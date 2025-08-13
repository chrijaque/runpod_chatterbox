import logging
import time
import redis

from .config import settings
from .services.runpod_client import RunPodClient
from .services.redis_queue import RedisQueueService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")


def get_redis_client() -> redis.Redis:
    if not settings.REDIS_URL:
        raise RuntimeError("REDIS_URL is not configured")
    return redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, ssl=settings.REDIS_USE_TLS)


def ensure_group(client: redis.Redis, stream: str, group: str) -> None:
    try:
        client.xgroup_create(stream, group, id="$", mkstream=True)
        logger.info(f"✅ Created consumer group '{group}' on '{stream}'")
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"ℹ️ Consumer group '{group}' already exists")
        else:
            raise


def main() -> None:
    # Use per-pod stream config so TTS and VC pods can run independently
    stream = settings.REDIS_STREAM_NAME
    group = settings.REDIS_CONSUMER_GROUP
    consumer = settings.REDIS_CONSUMER_NAME

    client = get_redis_client()
    ensure_group(client, stream, group)

    runpod = RunPodClient(
        api_key=settings.RUNPOD_API_KEY,
        voice_endpoint_id=settings.VC_CB_ENDPOINT_ID,
        tts_endpoint_id=settings.TTS_CB_ENDPOINT_ID,
    )

    in_flight_vc: int = 0
    in_flight_tts: int = 0
    max_vc: int = settings.RUNPOD_MAX_CONCURRENCY_VC
    max_tts: int = settings.RUNPOD_MAX_CONCURRENCY_TTS

    logger.info(
        f"👷 Worker started stream={stream} group={group} consumer={consumer} max_vc={max_vc} max_tts={max_tts}"
    )

    while True:
        try:
            # Choose which type we can take next based on available slots
            want_vc = in_flight_vc < max_vc
            want_tts = in_flight_tts < max_tts
            if not (want_vc or want_tts):
                time.sleep(0.2)
                continue

            messages = client.xreadgroup(group, consumer, streams={stream: ">"}, count=1, block=5000)
            if not messages:
                continue

            for _, entries in messages:
                for msg_id, data in entries:
                    job_id = data.get("job_id")
                    job_type = data.get("type")
                    logger.info(f"📥 Received job {job_id} type={job_type} msg_id={msg_id}")

                    try:
                        if job_type == "vc":
                            if in_flight_vc >= max_vc:
                                # skip processing now; re-add to stream tail
                                logger.info("⏭️ VC slot full, requeueing message")
                                client.xadd(stream, data)
                                client.xack(stream, group, msg_id)
                                continue
                            in_flight_vc += 1
                            name = data.get("payload:name")
                            audio_b64 = data.get("payload:audio_base64")
                            audio_format = data.get("payload:audio_format", "wav")
                            language = data.get("payload:language", "en")
                            is_kids = data.get("payload:is_kids_voice", "false").lower() == "true"
                            model_type = data.get("payload:model_type", "chatterbox")

                            result = runpod.create_voice_clone(
                                name=name,
                                audio_base64=audio_b64,
                                audio_format=audio_format,
                                language=language,
                                is_kids_voice=is_kids,
                                model_type=model_type,
                            )
                            logger.info(f"✅ VC job done {job_id}: status={result.get('status')}")
                        elif job_type == "tts":
                            if in_flight_tts >= max_tts:
                                logger.info("⏭️ TTS slot full, requeueing message")
                                client.xadd(stream, data)
                                client.xack(stream, group, msg_id)
                                continue
                            in_flight_tts += 1
                            voice_id = data.get("payload:voice_id")
                            text = data.get("payload:text")
                            profile_b64 = data.get("payload:profile_base64")
                            language = data.get("payload:language", "en")
                            story_type = data.get("payload:story_type", "user")
                            is_kids = data.get("payload:is_kids_voice", "false").lower() == "true"
                            model_type = data.get("payload:model_type", "chatterbox")

                            result = runpod.generate_tts_with_context(
                                voice_id=voice_id,
                                text=text,
                                profile_base64=profile_b64,
                                language=language,
                                story_type=story_type,
                                is_kids_voice=is_kids,
                                model_type=model_type,
                            )
                            logger.info(f"✅ TTS job done {job_id}: status={result.get('status')}")
                        else:
                            logger.warning(f"⚠️ Unknown job type: {job_type}")

                        client.xack(stream, group, msg_id)
                        logger.info(f"🧾 ACK job msg_id={msg_id}")
                        try:
                            RedisQueueService().set_job_status(job_id, "completed")
                        except Exception:
                            pass
                    except Exception as e:
                        logger.error(f"❌ Job {job_id} failed: {e}")
                        # retry with backoff or send to DLQ
                        try:
                            rq = RedisQueueService()
                            retries = rq.increment_retry(job_id)
                            if retries > settings.REDIS_MAX_RETRIES:
                                rq.set_job_status(job_id, "failed", error=str(e))
                                rq.send_to_dlq({"job_id": job_id, "type": job_type or "?", "error": str(e)})
                                client.xack(stream, group, msg_id)
                                logger.info(f"🧾 DLQ and ACK msg_id={msg_id}")
                            else:
                                delay = settings.REDIS_RETRY_BASE_DELAY_SECONDS * (2 ** (retries - 1))
                                rq.set_job_status(job_id, "retrying", retries=str(retries), delay=str(delay))
                                # Requeue with delay
                                fields = {k: v for k, v in data.items()}
                                rq.delay_requeue(fields, delay_seconds=delay)
                                client.xack(stream, group, msg_id)
                                logger.info(f"🔁 Requeued job {job_id} with delay={delay}s and ACK old msg")
                        except Exception as re_err:
                            logger.error(f"❌ Retry handling failed: {re_err}")
                    finally:
                        if job_type == "vc" and in_flight_vc > 0:
                            in_flight_vc -= 1
                        if job_type == "tts" and in_flight_tts > 0:
                            in_flight_tts -= 1

        except Exception as loop_error:
            logger.error(f"❌ Worker loop error: {loop_error}")
            time.sleep(1)


if __name__ == "__main__":
    main()



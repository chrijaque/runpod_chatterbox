import logging
import time
import redis

from .config import settings
from .services.runpod_client import RunPodClient


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

    in_flight: int = 0
    max_concurrency: int = settings.RUNPOD_MAX_CONCURRENCY

    logger.info(
        f"👷 Worker started stream={stream} group={group} consumer={consumer} max_concurrency={max_concurrency}"
    )

    while True:
        try:
            if in_flight >= max_concurrency:
                time.sleep(0.5)
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
                        in_flight += 1
                        if job_type == "vc":
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
                    except Exception as e:
                        logger.error(f"❌ Job {job_id} failed: {e}")
                    finally:
                        in_flight -= 1

        except Exception as loop_error:
            logger.error(f"❌ Worker loop error: {loop_error}")
            time.sleep(1)


if __name__ == "__main__":
    main()



import logging
from typing import Dict, Any, Optional

import redis

from ..config import settings


logger = logging.getLogger(__name__)


class RedisQueueService:
    """Lightweight Redis Streams wrapper for job enqueueing and state."""

    def __init__(self, redis_url: Optional[str] = None):
        if redis_url is None:
            redis_url = settings.REDIS_URL
        if not redis_url:
            raise ValueError("REDIS_URL is not configured")

        # Upstash supports rediss and HTTPS REST; we use redis-py with rediss URL
        self.client = redis.Redis.from_url(redis_url, decode_responses=True, ssl=settings.REDIS_USE_TLS)
        # Default combined stream; per-type streams below
        self.stream = settings.REDIS_STREAM_NAME
        self.stream_vc = settings.REDIS_STREAM_NAME_VC
        self.stream_tts = settings.REDIS_STREAM_NAME_TTS
        self.group = settings.REDIS_CONSUMER_GROUP
        self.namespace = settings.REDIS_NAMESPACE

        self._ensure_stream_group()

    def _ensure_stream_group(self) -> None:
        try:
            # MKSTREAM creates stream if missing; create consumer group at end ($)
            self.client.xgroup_create(name=self.stream, groupname=self.group, id="$", mkstream=True)
            logger.info(f"✅ Created Redis consumer group '{self.group}' on stream '{self.stream}'")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"ℹ️ Redis consumer group '{self.group}' already exists")
            else:
                raise

    def enqueue_job(self, job_id: str, job_type: str, payload: Dict[str, Any]) -> str:
        fields: Dict[str, Any] = {
            "job_id": job_id,
            "type": job_type,
            "status": "pending",
            "retries": "0",
        }
        # Flatten payload (redis streams are string -> string)
        for key, value in payload.items():
            fields[f"payload:{key}"] = value if isinstance(value, str) else str(value)

        # Route to per-type stream if defined
        target_stream = self.stream
        if job_type == "vc" and self.stream_vc:
            target_stream = self.stream_vc
        elif job_type == "tts" and self.stream_tts:
            target_stream = self.stream_tts

        msg_id = self.client.xadd(target_stream, fields)

        # Write job hash for status lookups
        job_key = f"{self.namespace}:job:{job_id}"
        self.client.hset(job_key, mapping={
            "status": "pending",
            "type": job_type,
            "msg_id": msg_id,
            "retries": "0",
            **{k: v for k, v in fields.items() if k.startswith("payload:")},
        })
        # Optional TTL
        ttl = int(settings.REDIS_JOB_TTL_SECONDS)
        if ttl > 0:
            self.client.expire(job_key, ttl)

        logger.info(f"📤 Enqueued job {job_id} (type={job_type}) stream={target_stream} msg_id={msg_id}")
        return msg_id

    def set_job_status(self, job_id: str, status: str, **extra: Any) -> None:
        job_key = f"{self.namespace}:job:{job_id}"
        mapping = {"status": status}
        for k, v in extra.items():
            mapping[k] = v if isinstance(v, str) else str(v)
        self.client.hset(job_key, mapping=mapping)

    def increment_retry(self, job_id: str) -> int:
        job_key = f"{self.namespace}:job:{job_id}"
        retries = self.client.hincrby(job_key, "retries", 1)
        return int(retries)

    def send_to_dlq(self, msg_data: Dict[str, Any]) -> None:
        dlq_stream = settings.REDIS_DLP_STREAM
        self.client.xadd(dlq_stream, msg_data)

    def delay_requeue(self, fields: Dict[str, Any], delay_seconds: int) -> None:
        # naive delay: sleep and re-xadd (acceptable for now)
        import time
        time.sleep(max(0, delay_seconds))
        # Choose stream based on type
        t = fields.get("type")
        target_stream = self.stream
        if t == "vc" and self.stream_vc:
            target_stream = self.stream_vc
        elif t == "tts" and self.stream_tts:
            target_stream = self.stream_tts
        self.client.xadd(target_stream, fields)

    def get_job(self, job_id: str) -> Dict[str, str]:
        job_key = f"{self.namespace}:job:{job_id}"
        return self.client.hgetall(job_key)



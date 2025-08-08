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
        self.stream = settings.REDIS_STREAM_NAME
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
        }
        # Flatten payload (redis streams are string -> string)
        for key, value in payload.items():
            fields[f"payload:{key}"] = value if isinstance(value, str) else str(value)

        msg_id = self.client.xadd(self.stream, fields)

        # Write job hash for status lookups
        job_key = f"{self.namespace}:job:{job_id}"
        self.client.hset(job_key, mapping={
            "status": "pending",
            "type": job_type,
            **{k: v for k, v in fields.items() if k.startswith("payload:")},
        })
        # Optional TTL
        ttl = int(settings.REDIS_JOB_TTL_SECONDS)
        if ttl > 0:
            self.client.expire(job_key, ttl)

        logger.info(f"📤 Enqueued job {job_id} (type={job_type}) msg_id={msg_id}")
        return msg_id

    def set_job_status(self, job_id: str, status: str, **extra: Any) -> None:
        job_key = f"{self.namespace}:job:{job_id}"
        mapping = {"status": status}
        for k, v in extra.items():
            mapping[k] = v if isinstance(v, str) else str(v)
        self.client.hset(job_key, mapping=mapping)

    def get_job(self, job_id: str) -> Dict[str, str]:
        job_key = f"{self.namespace}:job:{job_id}"
        return self.client.hgetall(job_key)



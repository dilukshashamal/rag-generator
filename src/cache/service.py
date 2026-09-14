import hashlib
import json
from typing import Any
from uuid import UUID

from redis import Redis


class Cache:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    @staticmethod
    def key(collection_id: UUID, kind: str, *parts: object) -> str:
        digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
        return f"rag:{collection_id}:{kind}:{digest}"

    def get(self, key: str) -> Any | None:
        value = self.redis.get(key)
        return json.loads(value) if value is not None else None

    def set(self, key: str, value: Any, ttl: int) -> None:
        self.redis.set(key, json.dumps(value), ex=ttl)

    def clear_collection(self, collection_id: UUID) -> None:
        for key in self.redis.scan_iter(match=f"rag:{collection_id}:*", count=100):
            self.redis.delete(key)

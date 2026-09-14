import hashlib
from uuid import UUID

from redis import Redis

from src.security.questions import ValidationError


class RateLimiter:
    """Atomic fixed windows; Redis failure fails closed at the service boundary."""

    def __init__(self, redis: Redis, window: int) -> None:
        self.redis = redis
        self.window = window

    def check(self, collection_id: UUID, principal: str, operation: str, limit: int) -> None:
        identity = hashlib.sha256(principal.encode()).hexdigest()
        key = f"rate:{collection_id}:{operation}:{identity}"
        count = self.redis.eval(
            "local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],ARGV[1]) end; return n",
            1, key, self.window,
        )
        if int(count) > limit:
            raise ValidationError("Too many requests. Please wait a minute and try again.")

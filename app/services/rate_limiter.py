"""
Rate limiting service for API protection and abuse prevention.
"""

import time
import asyncio
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import redis.asyncio as redis
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

# Global rate limiter instance
sliding_window_limiter: Optional["SlidingWindowRateLimiter"] = None

async def init_rate_limiters():
    """Initialize rate limiters."""
    global sliding_window_limiter
    if sliding_window_limiter is None:
        sliding_window_limiter = SlidingWindowRateLimiter(
            default_limit=RateLimit(
                requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
                burst_limit=settings.RATE_LIMIT_BURST
            )
        )
        logger.info("Rate limiters initialized")

@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests_per_minute: int
    burst_limit: int
    window_size: int = 60  # seconds

@dataclass
class RateLimitStatus:
    """Rate limit status response."""
    allowed: bool
    requests_remaining: int
    reset_time: float
    retry_after: Optional[int] = None

class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_windows: Dict[str, deque] = defaultdict(deque)
        self.rate_limits: Dict[str, RateLimit] = {}
        
    def configure_rate_limit(self, identifier: str, rate_limit: RateLimit):
        """Configure rate limit for a specific identifier."""
        self.rate_limits[identifier] = rate_limit
        
    async def is_allowed(self, identifier: str, rate_limit: Optional[RateLimit] = None) -> RateLimitStatus:
        """Check if request is allowed under rate limit."""
        if not settings.RATE_LIMIT_ENABLED:
            return RateLimitStatus(
                allowed=True,
                requests_remaining=float('inf'),
                reset_time=time.time() + 60
            )
            
        # Use provided rate limit or get configured one
        limit = rate_limit or self.rate_limits.get(identifier)
        if not limit:
            # Default rate limit
            limit = RateLimit(
                requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
                burst_limit=settings.RATE_LIMIT_BURST
            )
            
        if self.redis_client:
            return await self._check_redis_rate_limit(identifier, limit)
        else:
            return await self._check_local_rate_limit(identifier, limit)
            
    async def _check_redis_rate_limit(self, identifier: str, limit: RateLimit) -> RateLimitStatus:
        """Check rate limit using Redis backend."""
        current_time = time.time()
        window_start = current_time - limit.window_size
        
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(f"rate_limit:{identifier}", 0, window_start)
        
        # Count current requests
        pipe.zcard(f"rate_limit:{identifier}")
        
        # Add current request
        pipe.zadd(f"rate_limit:{identifier}", {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(f"rate_limit:{identifier}", limit.window_size + 1)
        
        results = await pipe.execute()
        current_requests = results[1]
        
        # Check burst limit first
        if current_requests >= limit.burst_limit:
            retry_after = int(limit.window_size - (current_time % limit.window_size))
            return RateLimitStatus(
                allowed=False,
                requests_remaining=0,
                reset_time=current_time + retry_after,
                retry_after=retry_after
            )
            
        # Check per-minute limit
        if current_requests >= limit.requests_per_minute:
            retry_after = int(limit.window_size - (current_time % limit.window_size))
            return RateLimitStatus(
                allowed=False,
                requests_remaining=0,
                reset_time=current_time + retry_after,
                retry_after=retry_after
            )
            
        return RateLimitStatus(
            allowed=True,
            requests_remaining=limit.requests_per_minute - current_requests - 1,
            reset_time=current_time + limit.window_size
        )
        
    async def _check_local_rate_limit(self, identifier: str, limit: RateLimit) -> RateLimitStatus:
        """Check rate limit using local memory (for development)."""
        current_time = time.time()
        window = self.local_windows[identifier]
        
        # Remove old entries
        while window and window[0] <= current_time - limit.window_size:
            window.popleft()
            
        # Check limits
        current_requests = len(window)
        
        if current_requests >= limit.burst_limit:
            retry_after = int(limit.window_size)
            return RateLimitStatus(
                allowed=False,
                requests_remaining=0,
                reset_time=current_time + retry_after,
                retry_after=retry_after
            )
            
        if current_requests >= limit.requests_per_minute:
            retry_after = int(limit.window_size)
            return RateLimitStatus(
                allowed=False,
                requests_remaining=0,
                reset_time=current_time + retry_after,
                retry_after=retry_after
            )
            
        # Add current request
        window.append(current_time)
        
        return RateLimitStatus(
            allowed=True,
            requests_remaining=limit.requests_per_minute - current_requests - 1,
            reset_time=current_time + limit.window_size
        )

class TokenBucketRateLimiter:
    """Token bucket rate limiter for burst handling."""
    
    @dataclass
    class Bucket:
        tokens: float
        last_refill: float
        capacity: float
        refill_rate: float  # tokens per second
        
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_buckets: Dict[str, TokenBucketRateLimiter.Bucket] = {}
        
    async def consume_token(
        self, 
        identifier: str, 
        tokens_requested: float = 1.0,
        capacity: float = 100.0,
        refill_rate: float = 10.0  # tokens per second
    ) -> Tuple[bool, float]:
        """Consume tokens from bucket. Returns (allowed, tokens_remaining)."""
        
        if self.redis_client:
            return await self._consume_redis_token(
                identifier, tokens_requested, capacity, refill_rate
            )
        else:
            return await self._consume_local_token(
                identifier, tokens_requested, capacity, refill_rate
            )
    
    async def _consume_local_token(
        self, 
        identifier: str, 
        tokens_requested: float,
        capacity: float,
        refill_rate: float
    ) -> Tuple[bool, float]:
        """Consume tokens from local bucket."""
        current_time = time.time()
        
        if identifier not in self.local_buckets:
            self.local_buckets[identifier] = self.Bucket(
                tokens=capacity,
                last_refill=current_time,
                capacity=capacity,
                refill_rate=refill_rate
            )
            
        bucket = self.local_buckets[identifier]
        
        # Refill tokens
        time_passed = current_time - bucket.last_refill
        new_tokens = time_passed * bucket.refill_rate
        bucket.tokens = min(bucket.capacity, bucket.tokens + new_tokens)
        bucket.last_refill = current_time
        
        # Check if we can consume tokens
        if bucket.tokens >= tokens_requested:
            bucket.tokens -= tokens_requested
            return True, bucket.tokens
        else:
            return False, bucket.tokens
            
    async def _consume_redis_token(
        self, 
        identifier: str, 
        tokens_requested: float,
        capacity: float,
        refill_rate: float
    ) -> Tuple[bool, float]:
        """Consume tokens from Redis bucket."""
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local current_time = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or current_time
        
        -- Refill tokens
        local time_passed = current_time - last_refill
        local new_tokens = time_passed * refill_rate
        tokens = math.min(capacity, tokens + new_tokens)
        
        -- Check if we can consume
        local allowed = 0
        if tokens >= tokens_requested then
            tokens = tokens - tokens_requested
            allowed = 1
        end
        
        -- Update bucket
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
        redis.call('EXPIRE', key, 3600)  -- Expire after 1 hour
        
        return {allowed, tokens}
        """
        
        current_time = time.time()
        result = await self.redis_client.eval(
            lua_script,
            1,
            f"bucket:{identifier}",
            capacity,
            refill_rate,
            tokens_requested,
            current_time
        )
        
        allowed = bool(result[0])
        tokens_remaining = float(result[1])
        
        return allowed, tokens_remaining

# Global rate limiter instances
sliding_window_limiter = SlidingWindowRateLimiter()
token_bucket_limiter = TokenBucketRateLimiter()

async def init_rate_limiters():
    """Initialize rate limiters with Redis connection if available."""
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        
        global sliding_window_limiter, token_bucket_limiter
        sliding_window_limiter = SlidingWindowRateLimiter(redis_client)
        token_bucket_limiter = TokenBucketRateLimiter(redis_client)
        
        logger.info("Rate limiters initialized with Redis backend")
        
    except Exception as e:
        logger.warning(f"Failed to connect to Redis, using local rate limiters: {e}")
        # Keep local instances 
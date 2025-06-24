"""Cache service for API responses."""

import json
import hashlib
import logging
from typing import Any, Optional
from datetime import timedelta

try:
    import redis
    from redis.exceptions import ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    ConnectionError = Exception
    TimeoutError = Exception

from ..core.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Cache service with Redis backend and in-memory fallback."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: dict = {}
        self.default_ttl = 3600  # 1 hour
        
        # Try to connect to Redis
        self._init_redis()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available, using in-memory cache")
            return
            
        try:
            if settings.REDIS_URL:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            else:
                logger.warning("Redis URL not configured, using in-memory cache")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        data_str = json.dumps(data, sort_keys=True)
        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.default_ttl
            
            if self.redis_client:
                self.redis_client.setex(
                    key, 
                    ttl, 
                    json.dumps(value, default=str)
                )
                return True
            else:
                self.memory_cache[key] = value
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                return self.memory_cache.pop(key, None) is not None
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            else:
                # For memory cache, clear all keys containing pattern
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                return len(keys_to_delete)
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
            return 0
    
    def get_chat_cache_key(self, messages: list, model: str, **kwargs) -> str:
        """Generate cache key for chat completion."""
        cache_data = {
            "messages": messages,
            "model": model,
            **kwargs
        }
        return self._generate_key("chat", cache_data)
    
    def get_embedding_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for embedding."""
        cache_data = {
            "text": text,
            "model": model
        }
        return self._generate_key("embedding", cache_data)


# Global cache instance
cache_service = CacheService() 
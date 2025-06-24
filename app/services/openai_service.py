"""OpenAI API service with chat completions and embeddings."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse

from ..core.config import settings
from ..core.openai_config import (
    OpenAIModels, 
    OpenAILimits, 
    ChatCompletionRequest,
    EmbeddingRequest
)
from .cache_service import cache_service
# Rate limiting now handled by middleware

logger = logging.getLogger(__name__)


class OpenAIService:
    """Service for OpenAI API interactions."""
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.usage_stats = {
            "chat_requests": 0,
            "embedding_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured - client will not be initialized")
            return
        
        try:
            self.client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=OpenAILimits.REQUEST_TIMEOUT
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        for attempt in range(OpenAILimits.RETRY_ATTEMPTS):
            try:
                logger.info(f"Making API call to {func.__name__} with args: {args} and kwargs: {kwargs}")
                return await func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "timeout" in error_str:
                    if attempt == OpenAILimits.RETRY_ATTEMPTS - 1:
                        raise e
                    wait_time = OpenAILimits.RETRY_DELAY * (2 ** attempt)
                    error_type = "rate limit" if "rate limit" in error_str else "timeout"
                    logger.warning(f"{error_type} hit, retrying in {wait_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Unexpected error in OpenAI API call: {e}")
                    raise e
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = OpenAIModels.GPT_4_1_NANO,
        max_tokens: int = 500,
        temperature: float = 0.7,
        stream: bool = False,
        use_cache: bool = True
    ) -> ChatCompletion:
        """Create a chat completion."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Check cache first
        if use_cache and not stream:
            cache_key = cache_service.get_chat_cache_key(
                messages, model, max_tokens=max_tokens, temperature=temperature
            )
            cached_response = await cache_service.get(cache_key)
            if cached_response:
                logger.debug("Returning cached chat completion")
                return ChatCompletion(**cached_response)
        
        # Rate limiting handled by middleware
        
        try:
            # Make API call with retry logic
            response = await self._retry_with_backoff(
                self.client.chat.completions.create,
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            
            # Update usage stats
            self.usage_stats["chat_requests"] += 1
            if hasattr(response, 'usage') and response.usage:
                self.usage_stats["total_tokens"] += response.usage.total_tokens
            
            # Cache response if not streaming
            if use_cache and not stream:
                await cache_service.set(cache_key, response.model_dump(), ttl=3600)
            
            logger.info(f"Chat completion created successfully with model {model}")
            return response
            
        except Exception as e:
            logger.error(f"Error creating chat completion: {e}")
            raise e
    
    async def create_chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = OpenAIModels.GPT_4_1_NANO,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Create a streaming chat completion."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Rate limiting handled by middleware
        
        try:
            stream = await self._retry_with_backoff(
                self.client.chat.completions.create,
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
            
            # Update usage stats
            self.usage_stats["chat_requests"] += 1
            logger.info(f"Streaming chat completion created successfully with model {model}")
            
        except Exception as e:
            logger.error(f"Error creating streaming chat completion: {e}")
            raise e
    
    async def create_embedding(
        self,
        text: str,
        model: str = OpenAIModels.TEXT_EMBEDDING_3_SMALL,
        use_cache: bool = True
    ) -> List[float]:
        """Create an embedding for text."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Check cache first
        if use_cache:
            cache_key = cache_service.get_embedding_cache_key(text, model)
            cached_embedding = await cache_service.get(cache_key)
            if cached_embedding:
                logger.debug("Returning cached embedding")
                return cached_embedding
        
        # Rate limiting handled by middleware
        
        try:
            # Make API call with retry logic
            response = await self._retry_with_backoff(
                self.client.embeddings.create,
                input=text,
                model=model
            )
            
            embedding = response.data[0].embedding
            
            # Update usage stats
            self.usage_stats["embedding_requests"] += 1
            if hasattr(response, 'usage') and response.usage:
                self.usage_stats["total_tokens"] += response.usage.total_tokens
            
            # Cache embedding
            if use_cache:
                await cache_service.set(cache_key, embedding, ttl=86400)  # 24 hours
            
            logger.info(f"Embedding created successfully with model {model}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise e
    
    async def create_embeddings_batch(
        self,
        texts: List[str],
        model: str = OpenAIModels.TEXT_EMBEDDING_3_SMALL,
        use_cache: bool = True
    ) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), OpenAILimits.MAX_EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + OpenAILimits.MAX_EMBEDDING_BATCH_SIZE]
            
            # Check cache for each text in batch
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            if use_cache:
                for j, text in enumerate(batch):
                    cache_key = cache_service.get_embedding_cache_key(text, model)
                    cached_embedding = await cache_service.get(cache_key)
                    if cached_embedding:
                        batch_embeddings.append(cached_embedding)
                    else:
                        batch_embeddings.append(None)
                        uncached_texts.append(text)
                        uncached_indices.append(j)
            else:
                uncached_texts = batch
                uncached_indices = list(range(len(batch)))
                batch_embeddings = [None] * len(batch)
            
            # Get embeddings for uncached texts
            if uncached_texts:
                # Rate limiting handled by middleware
                
                try:
                    response = await self._retry_with_backoff(
                        self.client.embeddings.create,
                        input=uncached_texts,
                        model=model
                    )
                    
                    # Update usage stats
                    self.usage_stats["embedding_requests"] += 1
                    if hasattr(response, 'usage') and response.usage:
                        self.usage_stats["total_tokens"] += response.usage.total_tokens
                    
                    # Cache and store embeddings
                    for k, embedding_data in enumerate(response.data):
                        embedding = embedding_data.embedding
                        batch_index = uncached_indices[k]
                        batch_embeddings[batch_index] = embedding
                        
                        # Cache embedding
                        if use_cache:
                            cache_key = cache_service.get_embedding_cache_key(
                                uncached_texts[k], model
                            )
                            await cache_service.set(cache_key, embedding, ttl=86400)
                
                except Exception as e:
                    logger.error(f"Error creating batch embeddings: {e}")
                    raise e
            
            embeddings.extend(batch_embeddings)
        
        logger.info(f"Batch embeddings created successfully for {len(texts)} texts")
        return embeddings
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            **self.usage_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on OpenAI service."""
        if not self.client:
            return {
                "status": "unhealthy",
                "error": "OpenAI client not initialized"
            }
        
        try:
            # Test with a simple embedding request
            await self.create_embedding("health check", use_cache=False)
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global OpenAI service instance (lazy initialization)
_openai_service = None

def get_openai_service() -> OpenAIService:
    """Get or create the global OpenAI service instance."""
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service 
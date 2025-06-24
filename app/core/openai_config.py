"""OpenAI API configuration and constants."""

from typing import Dict, Any
from pydantic import BaseModel


class OpenAIModels:
    """OpenAI model configurations."""
    
    # Chat models
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4_1_MINI = "gpt-4.1-mini"
    
    # Embedding models
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


class OpenAILimits:
    """Rate limits and constraints for OpenAI API."""
    
    # Rate limits (requests per minute)
    CHAT_RPM = 3500  # GPT-4 Turbo
    EMBEDDING_RPM = 3000  # Embedding models
    
    # Token limits
    MAX_CHAT_TOKENS = 128000  # GPT-4 Turbo context window
    MAX_EMBEDDING_TOKENS = 8191  # Ada-002 limit
    
    # Batch sizes
    MAX_EMBEDDING_BATCH_SIZE = 100
    
    # Timeouts
    REQUEST_TIMEOUT = 60  # seconds
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1  # seconds


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions."""
    
    messages: list[Dict[str, str]]
    model: str = OpenAIModels.GPT_4_1_MINI
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False


class EmbeddingRequest(BaseModel):
    """Request model for embeddings."""
    
    input: str | list[str]
    model: str = OpenAIModels.TEXT_EMBEDDING_3_SMALL
    encoding_format: str = "float"


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""
    
    id: str
    object: str
    created: int
    model: str
    choices: list[Dict[str, Any]]
    usage: Dict[str, Any]  # Changed from Dict[str, int] to handle nested token details


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    
    object: str
    data: list[Dict[str, Any]]
    model: str
    usage: Dict[str, int] 
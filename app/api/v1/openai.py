"""OpenAI API endpoints."""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...core.openai_config import OpenAIModels, ChatCompletionRequest, EmbeddingRequest
from ...core.dependencies import get_openai_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/openai", tags=["OpenAI"])


# Request/Response Models
class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatCompletionRequestModel(BaseModel):
    """Chat completion request model."""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: str = Field(default=OpenAIModels.GPT_4_1_NANO, description="Model to use")
    max_tokens: int = Field(default=500, ge=1, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(default=False, description="Whether to stream the response")


class EmbeddingRequestModel(BaseModel):
    """Embedding request model."""
    input: str | List[str] = Field(..., description="Text or list of texts to embed")
    model: str = Field(default=OpenAIModels.TEXT_EMBEDDING_3_SMALL, description="Embedding model to use")


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]  # Changed from Dict[str, int] to handle nested token details


class EmbeddingResponse(BaseModel):
    """Embedding response model."""
    object: str
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, Any]  # Changed from Dict[str, int] to handle nested token details


class UsageStatsResponse(BaseModel):
    """Usage statistics response model."""
    chat_requests: int
    embedding_requests: int
    total_tokens: int
    total_cost: float
    timestamp: str
    rate_limiter_stats: Dict[str, Dict[str, int]]


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequestModel):
    """Create a chat completion."""
    try:
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Validate message roles
        valid_roles = {"user", "assistant", "system"}
        for msg in messages:
            if msg["role"] not in valid_roles:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid message role: {msg['role']}. Must be one of: {valid_roles}"
                )
        
        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in get_openai_service().create_chat_completion_stream(
                    messages=messages,
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                ):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Return regular response
            response = await get_openai_service().create_chat_completion(
                messages=messages,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False
            )
            
            return ChatCompletionResponse(
                id=response.id,
                object=response.object,
                created=response.created,
                model=response.model,
                choices=[choice.model_dump() for choice in response.choices],
                usage=response.usage.model_dump() if response.usage else {}
            )
    
    except ValueError as e:
        logger.error(f"Validation error in chat completion: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating chat completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequestModel):
    """Create embeddings for text input."""
    try:
        if isinstance(request.input, str):
            # Single text embedding
            embedding = await get_openai_service().create_embedding(
                text=request.input,
                model=request.model
            )
            
            return EmbeddingResponse(
                object="list",
                data=[{
                    "object": "embedding",
                    "index": 0,
                    "embedding": embedding
                }],
                model=request.model,
                usage={"prompt_tokens": len(request.input.split()), "total_tokens": len(request.input.split())}
            )
        else:
            # Batch embeddings
            embeddings = await get_openai_service().create_embeddings_batch(
                texts=request.input,
                model=request.model
            )
            
            return EmbeddingResponse(
                object="list",
                data=[{
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding
                } for i, embedding in enumerate(embeddings)],
                model=request.model,
                usage={
                    "prompt_tokens": sum(len(text.split()) for text in request.input),
                    "total_tokens": sum(len(text.split()) for text in request.input)
                }
            )
    
    except ValueError as e:
        logger.error(f"Validation error in embedding creation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/usage/stats", response_model=UsageStatsResponse)
async def get_usage_stats():
    """Get OpenAI API usage statistics."""
    try:
        stats = get_openai_service().get_usage_stats()
        return UsageStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check():
    """Perform health check on OpenAI service."""
    try:
        health_status = await get_openai_service().health_check()
        
        if health_status["status"] == "healthy":
            return health_status
        else:
            raise HTTPException(status_code=503, detail=health_status)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models")
async def list_available_models():
    """List available OpenAI models."""
    return {
        "chat_models": [
            OpenAIModels.GPT_4_TURBO,
            OpenAIModels.GPT_4,
            OpenAIModels.GPT_3_5_TURBO,
            OpenAIModels.GPT_4_1_NANO,
            OpenAIModels.GPT_4_1_MINI
        ],
        "embedding_models": [
            OpenAIModels.TEXT_EMBEDDING_ADA_002,
            OpenAIModels.TEXT_EMBEDDING_3_SMALL,
            OpenAIModels.TEXT_EMBEDDING_3_LARGE
        ]
    } 
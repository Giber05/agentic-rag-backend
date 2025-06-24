"""
Health check and status API endpoints.
"""

import time
from datetime import datetime
from fastapi import APIRouter
from ..models.base import HealthResponse, StatusResponse
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Track application start time for uptime calculation
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns:
        HealthResponse: Basic health status and uptime information
    """
    uptime = time.time() - _start_time
    
    logger.info("Health check requested", uptime=uptime)
    
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        uptime=uptime
    )


@router.get("/api/v1/status", response_model=StatusResponse)
async def api_status():
    """
    Detailed API status endpoint with configuration information.
    
    Returns:
        StatusResponse: Comprehensive API status and configuration
    """
    agents_enabled = {
        "query_rewriter": settings.QUERY_REWRITER_ENABLED,
        "context_decision": settings.CONTEXT_DECISION_ENABLED,
        "source_retrieval": settings.SOURCE_RETRIEVAL_ENABLED,
        "answer_generation": settings.ANSWER_GENERATION_ENABLED,
        "validation_refinement": settings.VALIDATION_REFINEMENT_ENABLED,
    }
    
    performance_metrics = {
        "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS,
        "request_timeout": settings.REQUEST_TIMEOUT,
        "vector_search_timeout": settings.VECTOR_SEARCH_TIMEOUT,
        "embedding_batch_size": settings.EMBEDDING_BATCH_SIZE,
    }
    
    logger.info("API status requested", agents_enabled=agents_enabled)
    
    return StatusResponse(
        api_version=settings.VERSION,
        project_name=settings.PROJECT_NAME,
        description=settings.DESCRIPTION,
        agents_enabled=agents_enabled,
        performance_metrics=performance_metrics
    ) 
"""
Health check and status API endpoints.
"""

import time
import asyncio
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from ..models.base import HealthResponse, StatusResponse
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Track application start time for uptime calculation
_start_time = time.time()


async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity."""
    try:
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            return {"status": "warning", "message": "Database not configured"}
        
        # Simple check that imports work
        from supabase import create_client
        client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        
        # Test basic connectivity with timeout
        response = await asyncio.wait_for(
            asyncio.create_task(asyncio.to_thread(lambda: client.table("documents").select("id").limit(1).execute())),
            timeout=5.0
        )
        
        return {"status": "healthy", "message": "Database connected"}
    except asyncio.TimeoutError:
        return {"status": "unhealthy", "message": "Database timeout"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Database error: {str(e)[:100]}"}


async def check_openai_health() -> Dict[str, Any]:
    """Check OpenAI API connectivity."""
    try:
        if not settings.OPENAI_API_KEY:
            return {"status": "warning", "message": "OpenAI not configured"}
        
        import openai
        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Simple test with timeout
        await asyncio.wait_for(
            client.models.list(),
            timeout=5.0
        )
        
        return {"status": "healthy", "message": "OpenAI connected"}
    except asyncio.TimeoutError:
        return {"status": "unhealthy", "message": "OpenAI timeout"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"OpenAI error: {str(e)[:100]}"}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    This is optimized for Railway's health checking system.
    
    Returns:
        HealthResponse: Basic health status and uptime information
    """
    uptime = time.time() - _start_time
    
    # For Railway, we want a fast, simple health check
    if settings.is_production:
        logger.debug("Production health check requested", uptime=uptime)
        return HealthResponse(
            status="healthy",
            version=settings.VERSION,
            uptime=uptime
        )
    
    # For development, include more detailed checks
    logger.info("Health check requested", uptime=uptime)
    
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        uptime=uptime
    )


@router.get("/health/detailed", response_model=Dict[str, Any])
async def detailed_health_check():
    """
    Detailed health check with dependency verification.
    Use this for monitoring dashboards and debugging.
    """
    uptime = time.time() - _start_time
    
    # Run health checks concurrently
    db_check, openai_check = await asyncio.gather(
        check_database_health(),
        check_openai_health(),
        return_exceptions=True
    )
    
    # Handle any exceptions from health checks
    if isinstance(db_check, Exception):
        db_check = {"status": "error", "message": str(db_check)}
    if isinstance(openai_check, Exception):
        openai_check = {"status": "error", "message": str(openai_check)}
    
    # Determine overall health
    dependencies_healthy = all(
        check.get("status") in ["healthy", "warning"] 
        for check in [db_check, openai_check]
    )
    
    overall_status = "healthy" if dependencies_healthy else "degraded"
    
    health_data = {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "uptime_seconds": uptime,
        "dependencies": {
            "database": db_check,
            "openai": openai_check
        },
        "configuration": {
            "agents_enabled": {
                "query_rewriter": settings.QUERY_REWRITER_ENABLED,
                "context_decision": settings.CONTEXT_DECISION_ENABLED,
                "source_retrieval": settings.SOURCE_RETRIEVAL_ENABLED,
                "answer_generation": settings.ANSWER_GENERATION_ENABLED,
                "validation_refinement": settings.VALIDATION_REFINEMENT_ENABLED,
            }
        }
    }
    
    logger.info("Detailed health check completed", 
                overall_status=overall_status, 
                db_status=db_check.get("status"),
                openai_status=openai_check.get("status"))
    
    # Return 503 if unhealthy (for monitoring systems)
    if overall_status not in ["healthy", "degraded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_data
        )
    
    return health_data


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
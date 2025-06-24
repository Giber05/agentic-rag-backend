"""
Main FastAPI application for the Agentic RAG AI Agent backend.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

from .core.config import settings
from .core.logging import configure_logging, get_logger
from .core.api_docs import API_METADATA, OPENAPI_TAGS
from .api.health import router as health_router
from .api.database import router as database_router
from .api.v1.openai import router as openai_router
from .api.v1.documents import router as documents_router
from .api.v1.documents_simple import router as documents_simple_router
from .api.v1.search import router as search_router
from .api.v1.agents import router as agents_router
from .api.v1.query_rewriter import router as query_rewriter_router
from .api.v1.context_decision import router as context_decision_router
from .api.v1.source_retrieval import router as source_retrieval_router
from .api.v1.answer_generation import router as answer_generation_router
from .api.v1.rag_pipeline import router as rag_pipeline_router
from .api.v1.analytics import router as analytics_router
from .api.v1.auth import router as auth_router
from .api.v1.supabase_auth import router as supabase_auth_router
from .models.base import ErrorResponse
from .core.security import SECURITY_HEADERS
from .services.rate_limiter import init_rate_limiters

# Configure logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan event handler."""
    # Startup
    logger.info(
        "Starting Agentic RAG AI Agent backend",
        version=settings.VERSION,
        debug=settings.DEBUG,
        security_enabled={
            "rate_limiting": settings.RATE_LIMIT_ENABLED,
            "api_key_auth": settings.API_KEY_ENABLED,
            "security_headers": settings.SECURITY_HEADERS_ENABLED,
        },
        agents_enabled={
            "query_rewriter": settings.QUERY_REWRITER_ENABLED,
            "context_decision": settings.CONTEXT_DECISION_ENABLED,
            "source_retrieval": settings.SOURCE_RETRIEVAL_ENABLED,
            "answer_generation": settings.ANSWER_GENERATION_ENABLED,
            "validation_refinement": settings.VALIDATION_REFINEMENT_ENABLED,
        }
    )
    
    # Initialize rate limiters
    await init_rate_limiters()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agentic RAG AI Agent backend")


# Create FastAPI application
app = FastAPI(
    title=API_METADATA["title"],
    description=API_METADATA["description"],
    version=API_METADATA["version"],
    contact=API_METADATA["contact"],
    license_info=API_METADATA["license_info"],
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else settings.BACKEND_CORS_ORIGINS,  # Allow all origins in debug mode
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add unique request ID to each request for tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add security headers
    if settings.SECURITY_HEADERS_ENABLED:
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(
        "Request processed",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error responses."""
    request_id = getattr(request.state, "request_id", None)
    
    logger.error(
        "HTTP exception occurred",
        request_id=request_id,
        status_code=exc.status_code,
        detail=exc.detail,
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail,
            request_id=request_id
        ).model_dump(mode='json')
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured error responses."""
    request_id = getattr(request.state, "request_id", None)
    
    logger.error(
        "Unexpected exception occurred",
        request_id=request_id,
        exception=str(exc),
        exception_type=type(exc).__name__,
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred. Please try again later.",
            request_id=request_id
        ).model_dump(mode='json')
    )


# Include routers
app.include_router(health_router, tags=["health"])
app.include_router(auth_router, prefix=f"{settings.API_V1_STR}/auth", tags=["authentication"])
app.include_router(supabase_auth_router, prefix=f"{settings.API_V1_STR}/supabase-auth", tags=["supabase-authentication"])
app.include_router(database_router, prefix=settings.API_V1_STR, tags=["database"])
app.include_router(openai_router, prefix=settings.API_V1_STR, tags=["openai"])
app.include_router(documents_router, prefix=settings.API_V1_STR, tags=["documents"])
app.include_router(documents_simple_router, prefix=settings.API_V1_STR, tags=["documents-simple"])
app.include_router(search_router, prefix=settings.API_V1_STR, tags=["search"])
app.include_router(agents_router, prefix=settings.API_V1_STR, tags=["agents"])
app.include_router(query_rewriter_router, prefix=settings.API_V1_STR, tags=["query-rewriter"])
app.include_router(context_decision_router, prefix=settings.API_V1_STR, tags=["context-decision"])
app.include_router(source_retrieval_router, prefix=settings.API_V1_STR, tags=["source-retrieval"])
app.include_router(answer_generation_router, prefix=settings.API_V1_STR, tags=["answer-generation"])
app.include_router(rag_pipeline_router, prefix=settings.API_V1_STR, tags=["rag-pipeline"])
app.include_router(analytics_router, prefix=settings.API_V1_STR, tags=["analytics"])





if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    ) 
from functools import lru_cache
from supabase import create_client, Client
from ..core.config import settings
from ..services.openai_service import get_openai_service as _get_openai_service
from ..services.document_processor import DocumentProcessor
from ..services.document_service import DocumentService
from ..services.cache_service import CacheService
from ..services.rate_limiter import sliding_window_limiter, RateLimit
from ..services.vector_search_service import VectorSearchService
from ..services.user_service import UserService
from ..services.email_service import EmailService
from ..agents.registry import AgentRegistry
from ..agents.coordinator import AgentCoordinator
from ..agents.metrics import AgentMetrics
from ..agents.query_rewriter import QueryRewritingAgent
from ..agents.context_decision import ContextDecisionAgent
from ..agents.source_retrieval import SourceRetrievalAgent
from ..agents.answer_generation import AnswerGenerationAgent
from fastapi import Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import time
from jose import JWTError, jwt
from .security import verify_token, TokenData, validate_api_key, sanitize_input
from .logging import get_logger

logger = get_logger(__name__)

# Security schemes
security = HTTPBearer(auto_error=False)

class AuthenticationError(HTTPException):
    """Custom authentication error."""
    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

class AuthorizationError(HTTPException):
    """Custom authorization error."""
    def __init__(self, detail: str = "Not enough permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )

class RateLimitError(HTTPException):
    """Custom rate limit error."""
    def __init__(self, retry_after: int, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)}
        )

async def get_client_ip(request: Request) -> str:
    """Get client IP address with proxy support."""
    # Check for real IP from proxy headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
        
    # Fallback to direct connection
    return request.client.host if request.client else "unknown"

async def rate_limit_dependency(
    request: Request,
    client_ip: str = Depends(get_client_ip)
) -> None:
    """Rate limiting dependency."""
    if not settings.RATE_LIMIT_ENABLED:
        return
        
    # Create identifier (IP + endpoint)
    endpoint = request.url.path
    identifier = f"{client_ip}:{endpoint}"
    
    # Check rate limit
    rate_limit_status = await sliding_window_limiter.is_allowed(identifier)
    
    if not rate_limit_status.allowed:
        logger.warning(
            "Rate limit exceeded",
            client_ip=client_ip,
            endpoint=endpoint,
            retry_after=rate_limit_status.retry_after
        )
        raise RateLimitError(
            retry_after=rate_limit_status.retry_after or 60,
            detail=f"Rate limit exceeded. Try again in {rate_limit_status.retry_after} seconds."
        )

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenData]:
    """Get current user from JWT token (optional)."""
    if not credentials:
        return None
        
    try:
        payload = verify_token(credentials.credentials, "access")
        if not payload:
            return None
            
        token_data = TokenData(
            user_id=payload.get("sub"),
            email=payload.get("email"),
            scopes=payload.get("scopes", [])
        )
        
        return token_data
        
    except Exception as e:
        logger.warning(f"Token validation failed: {e}")
        return None

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> TokenData:
    """Get current user from JWT token (required)."""
    if not credentials:
        raise AuthenticationError("Authorization header missing")
        
    try:
        payload = verify_token(credentials.credentials, "access")
        if not payload:
            raise AuthenticationError("Invalid token")
            
        token_data = TokenData(
            user_id=payload.get("sub"),
            email=payload.get("email"),
            scopes=payload.get("scopes", [])
        )
        
        if not token_data.user_id:
            raise AuthenticationError("Invalid token payload")
            
        return token_data
        
    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise AuthenticationError("Invalid token")
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise AuthenticationError("Authentication failed")

async def verify_api_key_dependency(
    x_api_key: Optional[str] = Header(None, alias=settings.API_KEY_HEADER)
) -> bool:
    """Verify API key from header."""
    if not settings.API_KEY_ENABLED:
        return True
        
    if not x_api_key:
        raise AuthenticationError("API key required")
        
    # Here you would validate against stored API keys
    # For now, we'll use a simple validation
    if x_api_key.startswith("sk-") and len(x_api_key) > 20:
        return True
        
    raise AuthenticationError("Invalid API key")

def require_scopes(*required_scopes: str):
    """Dependency factory for scope-based authorization."""
    async def _require_scopes(
        current_user: TokenData = Depends(get_current_user)
    ) -> TokenData:
        for scope in required_scopes:
            if scope not in current_user.scopes:
                raise AuthorizationError(
                    f"Operation requires '{scope}' scope"
                )
        return current_user
    
    return _require_scopes

def require_permissions(*permissions: str):
    """Dependency factory for permission-based authorization."""
    async def _require_permissions(
        current_user: TokenData = Depends(get_current_user)
    ) -> TokenData:
        # Here you would check user permissions against database
        # For now, we'll use a simple scope-based check
        for permission in permissions:
            permission_scope = f"permission:{permission}"
            if permission_scope not in current_user.scopes:
                raise AuthorizationError(
                    f"Operation requires '{permission}' permission"
                )
        return current_user
    
    return _require_permissions

async def validate_request_size(request: Request) -> None:
    """Validate request content length."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > settings.MAX_REQUEST_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request too large. Maximum size: {settings.MAX_REQUEST_SIZE} bytes"
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid content-length header"
            )

async def sanitize_query_params(request: Request) -> Dict[str, Any]:
    """Sanitize query parameters."""
    sanitized = {}
    for key, value in request.query_params.items():
        sanitized_key = sanitize_input(key)
        if isinstance(value, str):
            sanitized_value = sanitize_input(value)
            # Check length limits
            if len(sanitized_value) > settings.MAX_QUERY_LENGTH:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Query parameter '{sanitized_key}' too long"
                )
            sanitized[sanitized_key] = sanitized_value
        else:
            sanitized[sanitized_key] = value
    
    return sanitized

# Dependency combinations
async def security_dependencies(
    _rate_limit: None = Depends(rate_limit_dependency),
    _request_size: None = Depends(validate_request_size),
    current_user: Optional[TokenData] = Depends(get_current_user_optional)
) -> Optional[TokenData]:
    """Combined security dependencies (optional auth)."""
    return current_user

async def authenticated_security_dependencies(
    _rate_limit: None = Depends(rate_limit_dependency),
    _request_size: None = Depends(validate_request_size),
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """Combined security dependencies (required auth)."""
    return current_user

async def api_key_security_dependencies(
    _rate_limit: None = Depends(rate_limit_dependency),
    _request_size: None = Depends(validate_request_size),
    _api_key: bool = Depends(verify_api_key_dependency)
) -> None:
    """Combined security dependencies with API key."""
    pass

# Role-based access control
class RoleChecker:
    """Role-based access control checker."""
    
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: TokenData = Depends(get_current_user)) -> TokenData:
        user_roles = current_user.scopes
        
        # Check if user has any of the required roles
        if not any(role in user_roles for role in self.allowed_roles):
            raise AuthorizationError(
                f"Access denied. Required roles: {', '.join(self.allowed_roles)}"
            )
            
        return current_user

# Pre-defined role checkers
require_admin = RoleChecker(["admin"])
require_user = RoleChecker(["user", "admin"])
require_moderator = RoleChecker(["moderator", "admin"])

@lru_cache()
def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

@lru_cache()
def get_cache_service() -> CacheService:
    """Get cache service instance"""
    return CacheService()

@lru_cache()
def get_rate_limiter():
    """Get rate limiter instance"""
    return sliding_window_limiter

def get_openai_service():
    """Get OpenAI service instance"""
    return _get_openai_service()

@lru_cache()
def get_document_processor() -> DocumentProcessor:
    """Get document processor instance"""
    openai_service = get_openai_service()
    return DocumentProcessor(openai_service)

@lru_cache()
def get_document_service() -> DocumentService:
    """Get document service instance"""
    supabase_client = get_supabase_client()
    return DocumentService(supabase_client)

@lru_cache()
def get_vector_search_service() -> VectorSearchService:
    """Get vector search service instance"""
    supabase_client = get_supabase_client()
    openai_service = get_openai_service()
    return VectorSearchService(supabase_client, openai_service)

@lru_cache()
def get_agent_metrics() -> AgentMetrics:
    """Get agent metrics instance"""
    return AgentMetrics()

@lru_cache()
def get_agent_registry() -> AgentRegistry:
    """Get agent registry instance"""
    registry = AgentRegistry()
    
    # Register agent types
    registry.register_agent_type("query_rewriter", QueryRewritingAgent)
    registry.register_agent_type("context_decision", ContextDecisionAgent)
    registry.register_agent_type("source_retrieval", SourceRetrievalAgent)
    registry.register_agent_type("answer_generation", AnswerGenerationAgent)
    
    return registry

@lru_cache()
def get_agent_coordinator() -> AgentCoordinator:
    """Get agent coordinator instance"""
    registry = get_agent_registry()
    return AgentCoordinator(registry) 

@lru_cache()
def get_user_service() -> UserService:
    """Get user service instance"""
    return UserService()

@lru_cache()
def get_email_service() -> EmailService:
    """Get email service instance"""
    return EmailService() 
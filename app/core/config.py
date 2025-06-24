"""
Core configuration settings for the Agentic RAG AI Agent backend.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator
import os
from .openai_config import OpenAIModels

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Agentic RAG AI Agent"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Advanced Agentic RAG AI Agent with 5 specialized agents"
    
    # Server Configuration - Railway compatible
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", 8000))  # Railway sets PORT automatically
    DEBUG: bool = os.getenv("ENVIRONMENT", "development") != "production"
    RELOAD: bool = os.getenv("ENVIRONMENT", "development") != "production"
    
    # Production/Development Environment Detection
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-change-this-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Rate Limiting - Relaxed for production stability
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 200  # Increased for production
    RATE_LIMIT_BURST: int = 400
    
    # Security Headers
    SECURITY_HEADERS_ENABLED: bool = True
    
    # API Key Authentication
    API_KEY_ENABLED: bool = False  # Disabled by default for Railway
    API_KEY_HEADER: str = "X-API-Key"
    
    # Input Validation
    MAX_REQUEST_SIZE: int = 10_000_000  # 10MB
    MAX_QUERY_LENGTH: int = 10000
    
    # Allowed Hosts for Production - Railway compatible
    ALLOWED_HOSTS: List[str] = ["*"]  # Railway handles this at proxy level
    
    # Email Configuration
    EMAIL_VERIFICATION_ENABLED: bool = False
    EMAIL_HOST: Optional[str] = None
    EMAIL_PORT: int = 587
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    EMAIL_USE_TLS: bool = True
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # OAuth Configuration
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None
    
    # CORS Configuration - Production ready
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://localhost:4200",
        "https://*.railway.app",  # Allow Railway domains
        "https://*.vercel.app",   # Allow Vercel domains
        "https://*.netlify.app"   # Allow Netlify domains
    ]
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database Configuration (Supabase) - Required for Railway
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY") 
    SUPABASE_SERVICE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_KEY")
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    SUPABASE_JWT_SECRET: Optional[str] = os.getenv("SUPABASE_JWT_SECRET", "your-jwt-secret")
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # OpenAI Configuration - Required for Railway
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = OpenAIModels.GPT_4_1_NANO
    OPENAI_EMBEDDING_MODEL: str = OpenAIModels.TEXT_EMBEDDING_3_SMALL
    OPENAI_MAX_TOKENS: int = 500
    OPENAI_TEMPERATURE: float = 0.7
    
    # Redis Configuration - Railway compatible (optional)
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    
    # Logging Configuration - Production optimized
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if os.getenv("ENVIRONMENT") == "production" else "DEBUG")
    LOG_FORMAT: str = "json"
    
    # Agent Configuration
    QUERY_REWRITER_ENABLED: bool = True
    CONTEXT_DECISION_ENABLED: bool = True
    SOURCE_RETRIEVAL_ENABLED: bool = True
    ANSWER_GENERATION_ENABLED: bool = True
    VALIDATION_REFINEMENT_ENABLED: bool = True
    
    # Performance Configuration - Railway optimized
    MAX_CONCURRENT_REQUESTS: int = 50  # Conservative for Railway free tier
    REQUEST_TIMEOUT: int = 60  # Increased timeout for production
    VECTOR_SEARCH_TIMEOUT: int = 10  # Increased for stability
    EMBEDDING_BATCH_SIZE: int = 50  # Reduced for memory efficiency
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"
        
    @property 
    def database_url_sync(self) -> str:
        """Get synchronous database URL."""
        if self.DATABASE_URL:
            return self.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")
        return ""


# Global settings instance
settings = Settings() 
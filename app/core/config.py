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
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    RELOAD: bool = True
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 200
    
    # Security Headers
    SECURITY_HEADERS_ENABLED: bool = True
    
    # API Key Authentication
    API_KEY_ENABLED: bool = True
    API_KEY_HEADER: str = "X-API-Key"
    
    # Input Validation
    MAX_REQUEST_SIZE: int = 10_000_000  # 10MB
    MAX_QUERY_LENGTH: int = 10000
    
    # Allowed Hosts for Production
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    # Email Configuration
    EMAIL_VERIFICATION_ENABLED: bool = False
    EMAIL_HOST: Optional[str] = None
    EMAIL_PORT: int = 587
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    EMAIL_USE_TLS: bool = True
    FRONTEND_URL: str = "http://localhost:3000"
    
    # OAuth Configuration
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://localhost:4200"
    ]
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database Configuration (Supabase)
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_SERVICE_KEY: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None
    SUPABASE_JWT_SECRET: Optional[str] = "your-jwt-secret"  # TODO: Get from Supabase dashboard
    DATABASE_URL: Optional[str] = None
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = OpenAIModels.GPT_4_1_NANO
    OPENAI_EMBEDDING_MODEL: str = OpenAIModels.TEXT_EMBEDDING_3_SMALL
    OPENAI_MAX_TOKENS: int = 500
    OPENAI_TEMPERATURE: float = 0.7
    
    # Redis Configuration (optional)
    REDIS_URL: Optional[str] = None
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Agent Configuration
    QUERY_REWRITER_ENABLED: bool = True
    CONTEXT_DECISION_ENABLED: bool = True
    SOURCE_RETRIEVAL_ENABLED: bool = True
    ANSWER_GENERATION_ENABLED: bool = True
    VALIDATION_REFINEMENT_ENABLED: bool = True
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT: int = 30
    VECTOR_SEARCH_TIMEOUT: int = 5
    EMBEDDING_BATCH_SIZE: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings() 
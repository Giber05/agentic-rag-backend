"""
Base models for the Agentic RAG AI Agent backend.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class BaseAPIModel(BaseModel):
    """Base model for API requests and responses."""
    
    class Config:
        from_attributes = True
        use_enum_values = True
        validate_assignment = True
        populate_by_name = True


class BaseResponse(BaseAPIModel):
    """Base response model."""
    
    success: bool = True
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TimestampedModel(BaseAPIModel):
    """Base model with timestamp fields."""
    
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class IdentifiedModel(TimestampedModel):
    """Base model with ID and timestamp fields."""
    
    id: UUID = Field(default_factory=uuid4)


class HealthResponse(BaseAPIModel):
    """Health check response model."""
    
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    uptime: Optional[float] = None


class StatusResponse(BaseAPIModel):
    """API status response model."""
    
    api_version: str
    project_name: str
    description: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agents_enabled: dict
    performance_metrics: Optional[dict] = None


class ErrorResponse(BaseAPIModel):
    """Error response model."""
    
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None 
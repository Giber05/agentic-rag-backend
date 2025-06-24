"""
Pydantic models for RAG Pipeline API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class BaseAPIModel(BaseModel):
    """Base model for API requests and responses."""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RAGRequest(BaseAPIModel):
    """Standard RAG request model."""
    query: str = Field(..., description="User query to process", min_length=1, max_length=2000)
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation messages")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Additional user context")
    pipeline_config: Optional[Dict[str, Any]] = Field(None, description="Pipeline configuration overrides")


class ProcessingResult(BaseAPIModel):
    """Result model for RAG processing."""
    request_id: str = Field(..., description="Unique request identifier")
    query: str = Field(..., description="Original user query")
    status: str = Field(..., description="Processing status")
    pipeline_type: str = Field(..., description="Type of pipeline used (optimized/full)")
    final_response: Optional[Dict[str, Any]] = Field(None, description="Generated response with citations")
    stage_results: Dict[str, Any] = Field(default_factory=dict, description="Results from each stage")
    total_duration: float = Field(..., description="Total processing time in seconds")
    optimization_info: Optional[Dict[str, Any]] = Field(None, description="Optimization and cost information")


class RAGProcessRequest(BaseAPIModel):
    """Request model for RAG pipeline processing."""
    query: str = Field(..., description="User query to process through RAG pipeline", min_length=1, max_length=2000)
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation messages for context")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Additional user context information")
    pipeline_config: Optional[Dict[str, Any]] = Field(None, description="Pipeline-specific configuration overrides")


class RAGStreamRequest(BaseAPIModel):
    """Request model for RAG pipeline streaming."""
    query: str = Field(..., description="User query to process through RAG pipeline", min_length=1, max_length=2000)
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation messages for context")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Additional user context information")
    pipeline_config: Optional[Dict[str, Any]] = Field(None, description="Pipeline-specific configuration overrides")


class RAGProcessResponse(BaseAPIModel):
    """Response model for RAG pipeline processing."""
    request_id: str = Field(..., description="Unique identifier for the pipeline request")
    query: str = Field(..., description="Original user query")
    status: str = Field(..., description="Pipeline execution status")
    final_response: Optional[Dict[str, Any]] = Field(None, description="Final generated response with citations")
    stage_results: Dict[str, Any] = Field(..., description="Results from each pipeline stage")
    metadata: Dict[str, Any] = Field(..., description="Pipeline execution metadata")
    total_duration: float = Field(..., description="Total pipeline execution time in seconds")
    error: Optional[str] = Field(None, description="Error message if pipeline failed")


class PipelineStatusResponse(BaseAPIModel):
    """Response model for pipeline status."""
    active_pipelines: int = Field(..., description="Number of currently active pipelines")
    cached_results: int = Field(..., description="Number of cached pipeline results")
    statistics: Dict[str, Any] = Field(..., description="Pipeline performance statistics")
    configuration: Dict[str, Any] = Field(..., description="Current pipeline configuration")
    timestamp: str = Field(..., description="Status timestamp")


class PipelineMetricsResponse(BaseAPIModel):
    """Response model for pipeline metrics."""
    total_pipelines: int = Field(..., description="Total number of pipelines processed")
    successful_pipelines: int = Field(..., description="Number of successful pipeline executions")
    failed_pipelines: int = Field(..., description="Number of failed pipeline executions")
    success_rate: float = Field(..., description="Pipeline success rate (0.0 to 1.0)")
    avg_duration: float = Field(..., description="Average pipeline execution time in seconds")
    stage_performance: Dict[str, Any] = Field(..., description="Performance metrics for each pipeline stage")
    active_pipelines: Dict[str, Any] = Field(..., description="Currently active pipeline information")
    cache_stats: Dict[str, Any] = Field(..., description="Cache performance statistics")


class ErrorResponse(BaseAPIModel):
    """Response model for API errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp") 
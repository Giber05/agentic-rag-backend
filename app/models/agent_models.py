"""
Pydantic models for agent framework API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from .base import BaseAPIModel


# Request Models

class CreateAgentRequest(BaseAPIModel):
    """Request model for creating a new agent."""
    agent_type: str = Field(..., description="Type of agent to create")
    agent_id: Optional[str] = Field(None, description="Optional specific ID for the agent")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")
    auto_start: bool = Field(True, description="Whether to start the agent automatically")


class ExecutePipelineRequest(BaseAPIModel):
    """Request model for executing the RAG pipeline."""
    query: str = Field(..., description="User query to process")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context data")


class QueryRewriteRequest(BaseAPIModel):
    """Request model for query rewriting."""
    query: str = Field(..., description="User query to rewrite and optimize", min_length=1, max_length=500)
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context data")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration overrides")


class ContextDecisionRequest(BaseAPIModel):
    """Request model for context decision evaluation."""
    query: str = Field(..., description="User query to evaluate for context necessity", min_length=1, max_length=500)
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation messages")
    current_context: Optional[Dict[str, Any]] = Field(None, description="Current context information")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration overrides")


class SourceRetrievalRequest(BaseAPIModel):
    """Request model for source retrieval."""
    query: str = Field(..., description="User query to retrieve sources for", min_length=1, max_length=500)
    context_decision: Optional[Dict[str, Any]] = Field(None, description="Context decision result from previous agent")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation messages")
    retrieval_config: Optional[Dict[str, Any]] = Field(None, description="Retrieval configuration overrides")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration overrides")


# Response Models

class AgentStateResponse(BaseAPIModel):
    """Response model for agent state information."""
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Type of agent")
    status: str = Field(..., description="Current agent status")
    created_at: datetime = Field(..., description="Agent creation time")
    started_at: Optional[datetime] = Field(None, description="Agent start time")
    stopped_at: Optional[datetime] = Field(None, description="Agent stop time")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional agent metadata")
    is_healthy: bool = Field(..., description="Whether the agent is in a healthy state")
    is_running: bool = Field(..., description="Whether the agent is currently running")


class AgentRegistryStatsResponse(BaseAPIModel):
    """Response model for agent registry statistics."""
    total_agents: int = Field(..., description="Total number of registered agents")
    registered_types: int = Field(..., description="Number of registered agent types")
    status_distribution: Dict[str, int] = Field(..., description="Distribution of agent statuses")
    type_distribution: Dict[str, int] = Field(..., description="Distribution of agent types")
    healthy_agents: int = Field(..., description="Number of healthy agents")
    running_agents: int = Field(..., description="Number of running agents")


class StepResultResponse(BaseAPIModel):
    """Response model for pipeline step result."""
    agent_id: str = Field(..., description="ID of the agent that executed the step")
    agent_type: str = Field(..., description="Type of the agent")
    success: bool = Field(..., description="Whether the step was successful")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    data: Optional[Dict[str, Any]] = Field(None, description="Step result data")
    error: Optional[str] = Field(None, description="Error message if unsuccessful")


class PipelineExecutionResponse(BaseAPIModel):
    """Response model for pipeline execution."""
    execution_id: str = Field(..., description="Unique execution identifier")
    query: str = Field(..., description="Original query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    status: str = Field(..., description="Execution status")
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    duration_ms: int = Field(..., description="Execution duration in milliseconds")
    current_step: Optional[str] = Field(None, description="Current pipeline step")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    step_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Results from each pipeline step")


class AgentMetricsResponse(BaseAPIModel):
    """Response model for agent performance metrics."""
    agent_id: str = Field(..., description="Agent identifier")
    agent_type: str = Field(..., description="Type of agent")
    total_operations: int = Field(..., description="Total number of operations")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    success_rate: float = Field(..., description="Success rate (0.0 to 1.0)")
    error_rate: float = Field(..., description="Error rate (0.0 to 1.0)")
    average_processing_time_ms: float = Field(..., description="Average processing time in milliseconds")
    min_processing_time_ms: float = Field(..., description="Minimum processing time in milliseconds")
    max_processing_time_ms: float = Field(..., description="Maximum processing time in milliseconds")
    throughput_per_minute: float = Field(..., description="Operations per minute")
    last_operation_time: Optional[datetime] = Field(None, description="Last operation timestamp")


class SystemOverviewResponse(BaseAPIModel):
    """Response model for system-wide metrics overview."""
    total_agents: int = Field(..., description="Total number of agents")
    total_operations: int = Field(..., description="Total operations across all agents")
    successful_operations: int = Field(..., description="Total successful operations")
    failed_operations: int = Field(..., description="Total failed operations")
    system_success_rate: float = Field(..., description="System-wide success rate")
    system_error_rate: float = Field(..., description="System-wide error rate")
    average_success_rate: float = Field(..., description="Average success rate across agents")
    average_processing_time_ms: float = Field(..., description="Average processing time across agents")
    total_throughput_per_minute: float = Field(..., description="Total system throughput per minute")
    agent_type_distribution: Dict[str, int] = Field(..., description="Distribution of agent types")
    active_agents: int = Field(..., description="Number of recently active agents")


class QueryRewriteResponse(BaseAPIModel):
    """Response model for query rewriting."""
    original_query: str = Field(..., description="Original user query")
    rewritten_query: str = Field(..., description="Optimized and rewritten query")
    preprocessing_steps: List[str] = Field(..., description="List of preprocessing steps applied")
    improvements: List[str] = Field(..., description="List of improvements made")
    confidence: float = Field(..., description="Confidence score for the rewriting", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the processing")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    agent_id: str = Field(..., description="ID of the agent that processed the request")


class ContextDecisionResponse(BaseAPIModel):
    """Response model for context decision evaluation."""
    query: str = Field(..., description="Original query that was evaluated")
    decision: str = Field(..., description="Context necessity decision: required, optional, or not_needed")
    confidence: float = Field(..., description="Confidence score for the decision", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation of the decision reasoning")
    decision_factors: Dict[str, Any] = Field(..., description="Detailed breakdown of decision factors")
    recommendations: List[str] = Field(..., description="Actionable recommendations based on the decision")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the evaluation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    agent_id: str = Field(..., description="ID of the agent that processed the request")


class SourceRetrievalResponse(BaseAPIModel):
    """Response model for source retrieval."""
    query: str = Field(..., description="Original query that was processed")
    strategy_used: str = Field(..., description="Retrieval strategy that was used")
    sources: List[Dict[str, Any]] = Field(..., description="Retrieved sources with relevance scores")
    total_sources: int = Field(..., description="Total number of sources retrieved")
    retrieval_metadata: Dict[str, Any] = Field(..., description="Metadata about the retrieval process")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    agent_id: str = Field(..., description="ID of the agent that processed the request")


class AgentStatsResponse(BaseAPIModel):
    """Response model for agent-specific statistics."""
    agent_type: str = Field(..., description="Type of agent")
    agents_info: List[Dict[str, Any]] = Field(..., description="Information about agent instances")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    registry_stats: Dict[str, Any] = Field(..., description="Registry statistics")


class AnswerGenerationRequest(BaseAPIModel):
    """Request model for answer generation."""
    query: str = Field(..., description="User query to generate answer for", min_length=1, max_length=1000)
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved sources to use for answer generation")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation messages for context")
    generation_config: Optional[Dict[str, Any]] = Field(None, description="Generation configuration overrides")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration overrides")


class AnswerGenerationResponse(BaseAPIModel):
    """Response model for answer generation."""
    query: str = Field(..., description="Original query that was processed")
    response_content: str = Field(..., description="Generated response content with citations")
    citations: List[Dict[str, Any]] = Field(..., description="List of citations used in the response")
    quality_metrics: Dict[str, Any] = Field(..., description="Response quality assessment metrics")
    format_type: str = Field(..., description="Format type of the response (markdown, plain_text, etc.)")
    word_count: int = Field(..., description="Number of words in the response")
    character_count: int = Field(..., description="Number of characters in the response")
    sources_used: int = Field(..., description="Number of sources used for generation")
    generation_metadata: Dict[str, Any] = Field(..., description="Generation metadata and performance info")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    agent_id: str = Field(..., description="ID of the agent that processed the request")


class StreamingRequest(BaseAPIModel):
    """Request model for streaming answer generation."""
    query: str = Field(..., description="User query to generate answer for", min_length=1, max_length=1000)
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved sources to use for answer generation")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation messages for context")
    generation_config: Optional[Dict[str, Any]] = Field(None, description="Generation configuration overrides")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration overrides")


class ErrorResponse(BaseAPIModel):
    """Response model for API errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp") 
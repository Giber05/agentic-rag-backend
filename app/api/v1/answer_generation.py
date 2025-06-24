"""
API endpoints for the Answer Generation Agent.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime

from ...agents.answer_generation import AnswerGenerationAgent
from ...agents.registry import AgentRegistry
from ...agents.metrics import AgentMetrics
from ...core.dependencies import get_agent_registry, get_agent_metrics
from ...models.agent_models import (
    AnswerGenerationRequest,
    AnswerGenerationResponse,
    StreamingRequest,
    AgentStatsResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/answer-generation", tags=["Answer Generation Agent"])


@router.post(
    "/generate",
    response_model=AnswerGenerationResponse,
    summary="Generate answer with citations",
    description="Generate a comprehensive answer with proper source citations based on retrieved context"
)
async def generate_answer(
    request: AnswerGenerationRequest,
    registry: AgentRegistry = Depends(get_agent_registry),
    metrics: AgentMetrics = Depends(get_agent_metrics)
) -> AnswerGenerationResponse:
    """
    Generate a comprehensive answer with citations based on retrieved sources.
    
    This endpoint:
    - Generates responses using LLM with retrieved context
    - Includes proper source citations in multiple styles
    - Provides response quality assessment
    - Supports multiple response formats
    - Includes performance metrics and metadata
    """
    try:
        # Get or create an answer generation agent
        agents = registry.get_agents_by_type("answer_generation")
        
        if not agents:
            # Create a new agent if none exists
            agent = await registry.create_agent(
                agent_type="answer_generation",
                agent_id="default_answer_generation",
                config=request.config,
                auto_start=True
            )
        else:
            # Use the first available healthy agent
            agent = None
            for candidate in agents:
                if candidate.is_healthy:
                    agent = candidate
                    break
            
            if not agent:
                raise HTTPException(
                    status_code=503,
                    detail="No healthy answer generation agents available"
                )
        
        # Process the generation request
        input_data = {
            "query": request.query,
            "sources": request.sources or [],
            "conversation_history": request.conversation_history or [],
            "generation_config": request.generation_config or {}
        }
        
        result = await agent.process(input_data)
        
        # Record metrics
        metrics.record_operation(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            success=result.success,
            processing_time_ms=result.processing_time_ms,
            operation_type="answer_generation"
        )
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Answer generation failed: {result.error}"
            )
        
        response_data = result.data["response"]
        
        return AnswerGenerationResponse(
            query=result.data["query"],
            response_content=response_data["content"],
            citations=response_data["citations"],
            quality_metrics=response_data["quality"],
            format_type=response_data["format_type"],
            word_count=response_data["word_count"],
            character_count=response_data["character_count"],
            sources_used=result.data["sources_used"],
            generation_metadata=result.data["generation_metadata"],
            processing_time_ms=result.processing_time_ms,
            agent_id=result.agent_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/stream",
    summary="Stream answer generation",
    description="Generate answer with real-time streaming for better user experience"
)
async def stream_answer(
    request: StreamingRequest,
    registry: AgentRegistry = Depends(get_agent_registry)
) -> StreamingResponse:
    """
    Stream answer generation in real-time.
    
    This endpoint provides:
    - Real-time response streaming
    - Progressive content delivery
    - Better user experience for long responses
    - Server-sent events format
    """
    try:
        # Get or create an answer generation agent
        agents = registry.get_agents_by_type("answer_generation")
        
        if not agents:
            agent = await registry.create_agent(
                agent_type="answer_generation",
                agent_id="streaming_answer_generation",
                config=request.config,
                auto_start=True
            )
        else:
            agent = None
            for candidate in agents:
                if candidate.is_healthy:
                    agent = candidate
                    break
            
            if not agent:
                raise HTTPException(
                    status_code=503,
                    detail="No healthy answer generation agents available"
                )
        
        # Check if streaming is enabled
        if not agent.enable_streaming:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not enabled for this agent"
            )
        
        async def generate_stream():
            """Generate streaming response."""
            try:
                # Send initial metadata
                yield f"data: {json.dumps({'type': 'start', 'query': request.query})}\n\n"
                
                # Stream the response
                async for chunk in agent.stream_response(
                    query=request.query,
                    sources=request.sources or [],
                    conversation_history=request.conversation_history or [],
                    generation_config=request.generation_config or {}
                ):
                    chunk_data = {
                        'type': 'chunk',
                        'content': chunk,
                        'timestamp': str(datetime.utcnow())
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                
            except Exception as e:
                error_data = {
                    'type': 'error',
                    'error': str(e),
                    'timestamp': str(datetime.utcnow())
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming answer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/quality",
    response_model=AgentStatsResponse,
    summary="Get answer quality metrics",
    description="Retrieve quality metrics and generation statistics for answer generation agents"
)
async def get_answer_quality(
    agent_id: Optional[str] = None,
    registry: AgentRegistry = Depends(get_agent_registry),
    metrics: AgentMetrics = Depends(get_agent_metrics)
) -> AgentStatsResponse:
    """
    Get quality metrics for answer generation agents.
    
    Args:
        agent_id: Optional specific agent ID to get metrics for
        
    Returns:
        Agent quality metrics and generation statistics
    """
    try:
        if agent_id:
            # Get metrics for specific agent
            agent = registry.get_agent(agent_id)
            if not agent or agent.agent_type != "answer_generation":
                raise HTTPException(
                    status_code=404,
                    detail=f"Answer generation agent {agent_id} not found"
                )
            
            agent_metrics = metrics.get_agent_metrics(agent_id)
            agents_info = [agent.state.__dict__]
        else:
            # Get metrics for all answer generation agents
            agents = registry.get_agents_by_type("answer_generation")
            if not agents:
                raise HTTPException(
                    status_code=404,
                    detail="No answer generation agents found"
                )
            
            # Aggregate metrics for all agents
            agent_metrics = None
            total_operations = 0
            total_successful = 0
            total_processing_time = 0.0
            
            agents_info = []
            for agent in agents:
                agent_info = agent.state.__dict__
                # Add agent-specific stats
                agent_info['performance_stats'] = agent._get_performance_stats()
                agents_info.append(agent_info)
                
                individual_metrics = metrics.get_agent_metrics(agent.agent_id)
                if individual_metrics:
                    total_operations += individual_metrics.total_operations
                    total_successful += individual_metrics.successful_operations
                    total_processing_time += (
                        individual_metrics.total_processing_time_ms
                    )
            
            # Create aggregated metrics
            if total_operations > 0:
                from ...agents.metrics import AgentPerformanceMetrics
                agent_metrics = AgentPerformanceMetrics(
                    agent_id="aggregated",
                    agent_type="answer_generation",
                    total_operations=total_operations,
                    successful_operations=total_successful,
                    failed_operations=total_operations - total_successful,
                    total_processing_time_ms=total_processing_time,
                    last_operation_time=None,
                    error_rate=1.0 - (total_successful / total_operations)
                )
        
        return AgentStatsResponse(
            agent_type="answer_generation",
            agents_info=agents_info,
            metrics=agent_metrics.__dict__ if agent_metrics else None,
            registry_stats=registry.get_registry_stats()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting answer quality metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/agent/create",
    summary="Create a new answer generation agent",
    description="Create and optionally start a new answer generation agent instance"
)
async def create_agent(
    agent_id: str,
    config: Optional[Dict[str, Any]] = None,
    auto_start: bool = True,
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Create a new answer generation agent.
    
    Args:
        agent_id: Unique identifier for the new agent
        config: Optional configuration for the agent
        auto_start: Whether to automatically start the agent after creation
        
    Returns:
        Information about the created agent
    """
    try:
        # Check if agent already exists
        existing_agent = registry.get_agent(agent_id)
        if existing_agent:
            raise HTTPException(
                status_code=409,
                detail=f"Agent with ID {agent_id} already exists"
            )
        
        # Create the agent
        agent = await registry.create_agent(
            agent_type="answer_generation",
            agent_id=agent_id,
            config=config,
            auto_start=auto_start
        )
        
        return {
            "message": f"Answer generation agent {agent_id} created successfully",
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "status": agent.state.status,
            "auto_started": auto_start,
            "configuration": {
                "max_response_length": agent.max_response_length,
                "citation_style": agent.citation_style.value,
                "response_format": agent.response_format.value,
                "enable_streaming": agent.enable_streaming,
                "quality_threshold": agent.quality_threshold,
                "model_name": agent.model_name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/agent/{agent_id}/config",
    summary="Get agent configuration",
    description="Retrieve current configuration for a specific agent"
)
async def get_agent_config(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Get current configuration for an answer generation agent.
    
    Args:
        agent_id: ID of the agent to get configuration for
        
    Returns:
        Current agent configuration
    """
    try:
        agent = registry.get_agent(agent_id)
        if not agent or agent.agent_type != "answer_generation":
            raise HTTPException(
                status_code=404,
                detail=f"Answer generation agent {agent_id} not found"
            )
        
        return {
            "agent_id": agent_id,
            "configuration": {
                "max_response_length": agent.max_response_length,
                "min_response_length": agent.min_response_length,
                "citation_style": agent.citation_style.value,
                "response_format": agent.response_format.value,
                "enable_streaming": agent.enable_streaming,
                "quality_threshold": agent.quality_threshold,
                "max_citations": agent.max_citations,
                "enable_quality_assessment": agent.enable_quality_assessment,
                "temperature": agent.temperature,
                "model_name": agent.model_name
            },
            "performance_stats": agent._get_performance_stats()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.put(
    "/agent/{agent_id}/config",
    summary="Update agent configuration",
    description="Update configuration for a specific agent"
)
async def update_agent_config(
    agent_id: str,
    config: Dict[str, Any],
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Update configuration for an answer generation agent.
    
    Args:
        agent_id: ID of the agent to update
        config: New configuration values
        
    Returns:
        Updated agent configuration
    """
    try:
        agent = registry.get_agent(agent_id)
        if not agent or agent.agent_type != "answer_generation":
            raise HTTPException(
                status_code=404,
                detail=f"Answer generation agent {agent_id} not found"
            )
        
        # Update configuration values
        if "max_response_length" in config:
            agent.max_response_length = max(50, min(4000, int(config["max_response_length"])))
        
        if "min_response_length" in config:
            agent.min_response_length = max(10, min(500, int(config["min_response_length"])))
        
        if "citation_style" in config:
            from ...agents.answer_generation import CitationStyle
            try:
                agent.citation_style = CitationStyle(config["citation_style"])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid citation style: {config['citation_style']}"
                )
        
        if "response_format" in config:
            from ...agents.answer_generation import ResponseFormat
            try:
                agent.response_format = ResponseFormat(config["response_format"])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid response format: {config['response_format']}"
                )
        
        if "enable_streaming" in config:
            agent.enable_streaming = bool(config["enable_streaming"])
        
        if "quality_threshold" in config:
            agent.quality_threshold = max(0.0, min(1.0, float(config["quality_threshold"])))
        
        if "max_citations" in config:
            agent.max_citations = max(1, min(20, int(config["max_citations"])))
        
        if "enable_quality_assessment" in config:
            agent.enable_quality_assessment = bool(config["enable_quality_assessment"])
        
        if "temperature" in config:
            agent.temperature = max(0.0, min(2.0, float(config["temperature"])))
        
        if "model_name" in config:
            agent.model_name = str(config["model_name"])
        
        return {
            "agent_id": agent_id,
            "message": "Configuration updated successfully",
            "updated_configuration": {
                "max_response_length": agent.max_response_length,
                "min_response_length": agent.min_response_length,
                "citation_style": agent.citation_style.value,
                "response_format": agent.response_format.value,
                "enable_streaming": agent.enable_streaming,
                "quality_threshold": agent.quality_threshold,
                "max_citations": agent.max_citations,
                "enable_quality_assessment": agent.enable_quality_assessment,
                "temperature": agent.temperature,
                "model_name": agent.model_name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/citation-styles",
    summary="Get available citation styles",
    description="List all available citation styles and their descriptions"
)
async def get_citation_styles() -> Dict[str, Any]:
    """
    Get information about available citation styles.
    
    Returns:
        Dictionary of available citation styles with descriptions
    """
    from ...agents.answer_generation import CitationStyle
    
    styles = {
        CitationStyle.NUMBERED.value: {
            "name": "Numbered Citations",
            "description": "Uses numbered citations like [1], [2] with sources list at the end",
            "example": "This is a fact [1]. Another claim [2].",
            "best_for": "Academic and formal documents"
        },
        CitationStyle.BRACKETED.value: {
            "name": "Bracketed Citations",
            "description": "Uses source titles in brackets like [Title]",
            "example": "This is a fact [Research Paper]. Another claim [Study Report].",
            "best_for": "Informal documents and quick reference"
        },
        CitationStyle.FOOTNOTE.value: {
            "name": "Footnote Citations",
            "description": "Uses superscript numbers with footnotes at the end",
            "example": "This is a fact^1. Another claim^2.",
            "best_for": "Traditional academic writing"
        },
        CitationStyle.INLINE.value: {
            "name": "Inline Citations",
            "description": "Embeds full citation information inline",
            "example": "This is a fact ([Research Paper](url)). Another claim (Study Report).",
            "best_for": "Web content and hyperlinked documents"
        }
    }
    
    return {
        "available_styles": styles,
        "default_style": CitationStyle.NUMBERED.value,
        "recommendation": "Use 'numbered' for formal documents or 'inline' for web content"
    }


@router.get(
    "/response-formats",
    summary="Get available response formats",
    description="List all available response formats and their descriptions"
)
async def get_response_formats() -> Dict[str, Any]:
    """
    Get information about available response formats.
    
    Returns:
        Dictionary of available response formats with descriptions
    """
    from ...agents.answer_generation import ResponseFormat
    
    formats = {
        ResponseFormat.MARKDOWN.value: {
            "name": "Markdown",
            "description": "Formatted text with Markdown syntax for headers, lists, links, etc.",
            "best_for": "Documentation, web content, and rich text display"
        },
        ResponseFormat.PLAIN_TEXT.value: {
            "name": "Plain Text",
            "description": "Simple text without formatting",
            "best_for": "Simple applications and text-only environments"
        },
        ResponseFormat.HTML.value: {
            "name": "HTML",
            "description": "HTML formatted text with tags for styling",
            "best_for": "Web applications and rich HTML display"
        },
        ResponseFormat.JSON.value: {
            "name": "JSON",
            "description": "Structured JSON format with separate content and metadata",
            "best_for": "API integrations and structured data processing"
        }
    }
    
    return {
        "available_formats": formats,
        "default_format": ResponseFormat.MARKDOWN.value,
        "recommendation": "Use 'markdown' for most applications or 'json' for API integrations"
    } 
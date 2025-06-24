"""
API endpoints for the Context Decision Agent.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
import logging

from ...agents.context_decision import ContextDecisionAgent
from ...agents.registry import AgentRegistry
from ...agents.metrics import AgentMetrics
from ...core.dependencies import get_agent_registry, get_agent_metrics
from ...models.agent_models import (
    ContextDecisionRequest,
    ContextDecisionResponse,
    AgentStatsResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context-decision", tags=["Context Decision Agent"])


@router.post(
    "/evaluate",
    response_model=ContextDecisionResponse,
    summary="Evaluate context necessity for a query",
    description="Determine whether additional context retrieval is needed for a query"
)
async def evaluate_context_necessity(
    request: ContextDecisionRequest,
    registry: AgentRegistry = Depends(get_agent_registry),
    metrics: AgentMetrics = Depends(get_agent_metrics)
) -> ContextDecisionResponse:
    """
    Evaluate whether additional context retrieval is needed for a query.
    
    This endpoint:
    - Analyzes query patterns and conversation history
    - Performs semantic similarity assessment
    - Uses AI-powered decision making (if enabled)
    - Provides confidence scoring and reasoning
    - Returns actionable recommendations
    """
    try:
        # Get or create a context decision agent
        agents = registry.get_agents_by_type("context_decision")
        
        if not agents:
            # Create a new agent if none exists
            agent = await registry.create_agent(
                agent_type="context_decision",
                agent_id="default_context_decision",
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
                    detail="No healthy context decision agents available"
                )
        
        # Process the evaluation request
        input_data = {
            "query": request.query,
            "conversation_history": request.conversation_history or [],
            "current_context": request.current_context or {}
        }
        
        result = await agent.process(input_data)
        
        # Record metrics
        metrics.record_operation(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            success=result.success,
            processing_time_ms=result.processing_time_ms,
            operation_type="context_evaluation"
        )
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Context evaluation failed: {result.error}"
            )
        
        return ContextDecisionResponse(
            query=result.data["query"],
            decision=result.data["decision"],
            confidence=result.data["confidence"],
            reasoning=result.data["reasoning"],
            decision_factors=result.data["decision_factors"],
            recommendations=result.data["recommendations"],
            metadata=result.data["metadata"],
            processing_time_ms=result.processing_time_ms,
            agent_id=result.agent_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating context necessity: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/metrics",
    response_model=AgentStatsResponse,
    summary="Get context decision agent metrics",
    description="Retrieve performance metrics and decision statistics for context decision agents"
)
async def get_decision_metrics(
    agent_id: Optional[str] = None,
    registry: AgentRegistry = Depends(get_agent_registry),
    metrics: AgentMetrics = Depends(get_agent_metrics)
) -> AgentStatsResponse:
    """
    Get performance metrics for context decision agents.
    
    Args:
        agent_id: Optional specific agent ID to get metrics for
        
    Returns:
        Agent performance metrics and decision statistics
    """
    try:
        if agent_id:
            # Get metrics for specific agent
            agent = registry.get_agent(agent_id)
            if not agent or agent.agent_type != "context_decision":
                raise HTTPException(
                    status_code=404,
                    detail=f"Context decision agent {agent_id} not found"
                )
            
            agent_metrics = metrics.get_agent_metrics(agent_id)
            agents_info = [agent.state.__dict__]
        else:
            # Get metrics for all context decision agents
            agents = registry.get_agents_by_type("context_decision")
            if not agents:
                raise HTTPException(
                    status_code=404,
                    detail="No context decision agents found"
                )
            
            # Aggregate metrics for all agents
            agent_metrics = None
            total_operations = 0
            total_successful = 0
            total_processing_time = 0.0
            
            agents_info = []
            for agent in agents:
                agent_info = agent.state.__dict__
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
                    agent_type="context_decision",
                    total_operations=total_operations,
                    successful_operations=total_successful,
                    failed_operations=total_operations - total_successful,
                    total_processing_time_ms=total_processing_time,
                    last_operation_time=None,
                    error_rate=1.0 - (total_successful / total_operations)
                )
        
        return AgentStatsResponse(
            agent_type="context_decision",
            agents_info=agents_info,
            metrics=agent_metrics.__dict__ if agent_metrics else None,
            registry_stats=registry.get_registry_stats()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting decision metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/agent/create",
    summary="Create a new context decision agent",
    description="Create and optionally start a new context decision agent instance"
)
async def create_agent(
    agent_id: str,
    config: Optional[Dict[str, Any]] = None,
    auto_start: bool = True,
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Create a new context decision agent.
    
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
            agent_type="context_decision",
            agent_id=agent_id,
            config=config,
            auto_start=auto_start
        )
        
        return {
            "message": f"Context decision agent {agent_id} created successfully",
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "status": agent.state.status,
            "auto_started": auto_start
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
    "/agent/{agent_id}/thresholds",
    summary="Get agent decision thresholds",
    description="Retrieve current decision thresholds for a specific agent"
)
async def get_agent_thresholds(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Get current decision thresholds for a context decision agent.
    
    Args:
        agent_id: ID of the agent to get thresholds for
        
    Returns:
        Current threshold configuration
    """
    try:
        agent = registry.get_agent(agent_id)
        if not agent or agent.agent_type != "context_decision":
            raise HTTPException(
                status_code=404,
                detail=f"Context decision agent {agent_id} not found"
            )
        
        return {
            "agent_id": agent_id,
            "thresholds": {
                "similarity_threshold": agent.similarity_threshold,
                "min_confidence_threshold": agent.min_confidence_threshold,
                "context_window_size": agent.context_window_size
            },
            "configuration": {
                "enable_ai_assessment": agent.enable_ai_assessment,
                "adaptive_thresholds": agent.adaptive_thresholds
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent thresholds: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.put(
    "/agent/{agent_id}/thresholds",
    summary="Update agent decision thresholds",
    description="Update decision thresholds for a specific agent"
)
async def update_agent_thresholds(
    agent_id: str,
    thresholds: Dict[str, float],
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Update decision thresholds for a context decision agent.
    
    Args:
        agent_id: ID of the agent to update
        thresholds: New threshold values
        
    Returns:
        Updated threshold configuration
    """
    try:
        agent = registry.get_agent(agent_id)
        if not agent or agent.agent_type != "context_decision":
            raise HTTPException(
                status_code=404,
                detail=f"Context decision agent {agent_id} not found"
            )
        
        # Update thresholds
        if "similarity_threshold" in thresholds:
            agent.similarity_threshold = max(0.0, min(1.0, thresholds["similarity_threshold"]))
        
        if "min_confidence_threshold" in thresholds:
            agent.min_confidence_threshold = max(0.0, min(1.0, thresholds["min_confidence_threshold"]))
        
        if "context_window_size" in thresholds:
            agent.context_window_size = max(1, int(thresholds["context_window_size"]))
        
        return {
            "agent_id": agent_id,
            "message": "Thresholds updated successfully",
            "updated_thresholds": {
                "similarity_threshold": agent.similarity_threshold,
                "min_confidence_threshold": agent.min_confidence_threshold,
                "context_window_size": agent.context_window_size
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent thresholds: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) 
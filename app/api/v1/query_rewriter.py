"""
API endpoints for the Query Rewriting Agent.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import logging

from ...agents.query_rewriter import QueryRewritingAgent
from ...agents.registry import AgentRegistry
from ...agents.metrics import AgentMetrics
from ...core.dependencies import get_agent_registry, get_agent_metrics
from ...models.agent_models import (
    QueryRewriteRequest,
    QueryRewriteResponse,
    AgentStatsResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query-rewriter", tags=["Query Rewriter Agent"])


@router.post(
    "/process",
    response_model=QueryRewriteResponse,
    summary="Process and rewrite a query",
    description="Rewrite and optimize a user query for better search results"
)
async def process_query(
    request: QueryRewriteRequest,
    registry: AgentRegistry = Depends(get_agent_registry),
    metrics: AgentMetrics = Depends(get_agent_metrics)
) -> QueryRewriteResponse:
    """
    Process and rewrite a user query.
    
    This endpoint:
    - Validates the input query
    - Applies spell and grammar correction
    - Normalizes the query for consistent processing
    - Optionally expands the query with related terms
    - Returns the optimized query with metadata
    """
    try:
        # Get or create a query rewriter agent
        agents = registry.get_agents_by_type("query_rewriter")
        
        if not agents:
            # Create a new agent if none exists
            agent = await registry.create_agent(
                agent_type="query_rewriter",
                agent_id="default_query_rewriter",
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
                    detail="No healthy query rewriter agents available"
                )
        
        # Process the query
        input_data = {
            "query": request.query,
            "conversation_id": request.conversation_id,
            "context": request.context or {}
        }
        
        result = await agent.process(input_data)
        
        # Record metrics
        metrics.record_operation(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            success=result.success,
            processing_time_ms=result.processing_time_ms,
            operation_type="query_rewrite"
        )
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Query rewriting failed: {result.error}"
            )
        
        return QueryRewriteResponse(
            original_query=result.data["original_query"],
            rewritten_query=result.data["rewritten_query"],
            preprocessing_steps=result.data["preprocessing_steps"],
            improvements=result.data["improvements"],
            confidence=result.data["confidence"],
            metadata=result.data["metadata"],
            processing_time_ms=result.processing_time_ms,
            agent_id=result.agent_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=AgentStatsResponse,
    summary="Get query rewriter agent statistics",
    description="Retrieve performance statistics for query rewriter agents"
)
async def get_agent_stats(
    agent_id: Optional[str] = None,
    registry: AgentRegistry = Depends(get_agent_registry),
    metrics: AgentMetrics = Depends(get_agent_metrics)
) -> AgentStatsResponse:
    """
    Get performance statistics for query rewriter agents.
    
    Args:
        agent_id: Optional specific agent ID to get stats for
        
    Returns:
        Agent performance statistics and metrics
    """
    try:
        if agent_id:
            # Get stats for specific agent
            agent = registry.get_agent(agent_id)
            if not agent or agent.agent_type != "query_rewriter":
                raise HTTPException(
                    status_code=404,
                    detail=f"Query rewriter agent {agent_id} not found"
                )
            
            agent_metrics = metrics.get_agent_metrics(agent_id)
            agents_info = [agent.state.dict()]
        else:
            # Get stats for all query rewriter agents
            agents = registry.get_agents_by_type("query_rewriter")
            if not agents:
                raise HTTPException(
                    status_code=404,
                    detail="No query rewriter agents found"
                )
            
            # Aggregate metrics for all agents
            agent_metrics = None
            total_operations = 0
            total_successful = 0
            total_processing_time = 0.0
            
            agents_info = []
            for agent in agents:
                agent_info = agent.state.dict()
                agents_info.append(agent_info)
                
                individual_metrics = metrics.get_agent_metrics(agent.agent_id)
                if individual_metrics:
                    total_operations += individual_metrics.total_operations
                    total_successful += individual_metrics.successful_operations
                    total_processing_time += (
                        individual_metrics.average_processing_time_ms * 
                        individual_metrics.total_operations
                    )
            
            # Create aggregated metrics
            if total_operations > 0:
                from ...agents.metrics import AgentPerformanceMetrics
                agent_metrics = AgentPerformanceMetrics(
                    agent_id="aggregated",
                    agent_type="query_rewriter",
                    total_operations=total_operations,
                    successful_operations=total_successful,
                    failed_operations=total_operations - total_successful,
                    total_processing_time_ms=total_processing_time,
                    last_operation_time=None,
                    error_rate=1.0 - (total_successful / total_operations)
                )
        
        return AgentStatsResponse(
            agent_type="query_rewriter",
            agents_info=agents_info,
            metrics=agent_metrics.__dict__ if agent_metrics else None,
            registry_stats=registry.get_registry_stats()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/agent/create",
    summary="Create a new query rewriter agent",
    description="Create and optionally start a new query rewriter agent instance"
)
async def create_agent(
    agent_id: str,
    config: Optional[Dict[str, Any]] = None,
    auto_start: bool = True,
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Create a new query rewriter agent.
    
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
            agent_type="query_rewriter",
            agent_id=agent_id,
            config=config,
            auto_start=auto_start
        )
        
        return {
            "message": f"Query rewriter agent {agent_id} created successfully",
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
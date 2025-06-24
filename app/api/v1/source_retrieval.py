"""
API endpoints for the Source Retrieval Agent.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
import logging

from ...agents.source_retrieval import SourceRetrievalAgent
from ...agents.registry import AgentRegistry
from ...agents.metrics import AgentMetrics
from ...core.dependencies import get_agent_registry, get_agent_metrics
from ...models.agent_models import (
    SourceRetrievalRequest,
    SourceRetrievalResponse,
    AgentStatsResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/source-retrieval", tags=["Source Retrieval Agent"])


@router.post(
    "/retrieve",
    response_model=SourceRetrievalResponse,
    summary="Retrieve relevant sources for a query",
    description="Perform semantic search and retrieve relevant context from knowledge base"
)
async def retrieve_sources(
    request: SourceRetrievalRequest,
    registry: AgentRegistry = Depends(get_agent_registry),
    metrics: AgentMetrics = Depends(get_agent_metrics)
) -> SourceRetrievalResponse:
    """
    Retrieve relevant sources for a query using various search strategies.
    
    This endpoint:
    - Performs semantic search using vector embeddings
    - Supports hybrid search combining semantic and keyword matching
    - Applies dynamic source selection and ranking
    - Provides result deduplication and filtering
    - Returns relevance scores and metadata
    """
    try:
        # Get or create a source retrieval agent
        agents = registry.get_agents_by_type("source_retrieval")
        
        if not agents:
            # Create a new agent if none exists
            agent = await registry.create_agent(
                agent_type="source_retrieval",
                agent_id="default_source_retrieval",
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
                    detail="No healthy source retrieval agents available"
                )
        
        # Process the retrieval request
        input_data = {
            "query": request.query,
            "context_decision": request.context_decision or {},
            "conversation_history": request.conversation_history or [],
            "retrieval_config": request.retrieval_config or {}
        }
        
        result = await agent.process(input_data)
        
        # Record metrics
        metrics.record_operation(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            success=result.success,
            processing_time_ms=result.processing_time_ms,
            operation_type="source_retrieval"
        )
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Source retrieval failed: {result.error}"
            )
        
        return SourceRetrievalResponse(
            query=result.data["query"],
            strategy_used=result.data["strategy_used"],
            sources=result.data["sources"],
            total_sources=result.data["total_sources"],
            retrieval_metadata=result.data["retrieval_metadata"],
            processing_time_ms=result.processing_time_ms,
            agent_id=result.agent_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving sources: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/performance",
    response_model=AgentStatsResponse,
    summary="Get source retrieval performance metrics",
    description="Retrieve performance metrics and retrieval statistics for source retrieval agents"
)
async def get_retrieval_performance(
    agent_id: Optional[str] = None,
    registry: AgentRegistry = Depends(get_agent_registry),
    metrics: AgentMetrics = Depends(get_agent_metrics)
) -> AgentStatsResponse:
    """
    Get performance metrics for source retrieval agents.
    
    Args:
        agent_id: Optional specific agent ID to get metrics for
        
    Returns:
        Agent performance metrics and retrieval statistics
    """
    try:
        if agent_id:
            # Get metrics for specific agent
            agent = registry.get_agent(agent_id)
            if not agent or agent.agent_type != "source_retrieval":
                raise HTTPException(
                    status_code=404,
                    detail=f"Source retrieval agent {agent_id} not found"
                )
            
            agent_metrics = metrics.get_agent_metrics(agent_id)
            agents_info = [agent.state.__dict__]
        else:
            # Get metrics for all source retrieval agents
            agents = registry.get_agents_by_type("source_retrieval")
            if not agents:
                raise HTTPException(
                    status_code=404,
                    detail="No source retrieval agents found"
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
                    agent_type="source_retrieval",
                    total_operations=total_operations,
                    successful_operations=total_successful,
                    failed_operations=total_operations - total_successful,
                    total_processing_time_ms=total_processing_time,
                    last_operation_time=None,
                    error_rate=1.0 - (total_successful / total_operations)
                )
        
        return AgentStatsResponse(
            agent_type="source_retrieval",
            agents_info=agents_info,
            metrics=agent_metrics.__dict__ if agent_metrics else None,
            registry_stats=registry.get_registry_stats()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting retrieval performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/agent/create",
    summary="Create a new source retrieval agent",
    description="Create and optionally start a new source retrieval agent instance"
)
async def create_agent(
    agent_id: str,
    config: Optional[Dict[str, Any]] = None,
    auto_start: bool = True,
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Create a new source retrieval agent.
    
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
            agent_type="source_retrieval",
            agent_id=agent_id,
            config=config,
            auto_start=auto_start
        )
        
        return {
            "message": f"Source retrieval agent {agent_id} created successfully",
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
    "/agent/{agent_id}/config",
    summary="Get agent configuration",
    description="Retrieve current configuration for a specific agent"
)
async def get_agent_config(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry)
) -> Dict[str, Any]:
    """
    Get current configuration for a source retrieval agent.
    
    Args:
        agent_id: ID of the agent to get configuration for
        
    Returns:
        Current agent configuration
    """
    try:
        agent = registry.get_agent(agent_id)
        if not agent or agent.agent_type != "source_retrieval":
            raise HTTPException(
                status_code=404,
                detail=f"Source retrieval agent {agent_id} not found"
            )
        
        return {
            "agent_id": agent_id,
            "configuration": {
                "max_results": agent.max_results,
                "min_relevance_threshold": agent.min_relevance_threshold,
                "semantic_weight": agent.semantic_weight,
                "keyword_weight": agent.keyword_weight,
                "enable_deduplication": agent.enable_deduplication,
                "similarity_threshold": agent.similarity_threshold,
                "default_strategy": agent.default_strategy.value,
                "enable_multi_source": agent.enable_multi_source
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
    Update configuration for a source retrieval agent.
    
    Args:
        agent_id: ID of the agent to update
        config: New configuration values
        
    Returns:
        Updated agent configuration
    """
    try:
        agent = registry.get_agent(agent_id)
        if not agent or agent.agent_type != "source_retrieval":
            raise HTTPException(
                status_code=404,
                detail=f"Source retrieval agent {agent_id} not found"
            )
        
        # Update configuration values
        if "max_results" in config:
            agent.max_results = max(1, min(100, int(config["max_results"])))
        
        if "min_relevance_threshold" in config:
            agent.min_relevance_threshold = max(0.0, min(1.0, float(config["min_relevance_threshold"])))
        
        if "semantic_weight" in config:
            agent.semantic_weight = max(0.0, min(1.0, float(config["semantic_weight"])))
        
        if "keyword_weight" in config:
            agent.keyword_weight = max(0.0, min(1.0, float(config["keyword_weight"])))
        
        if "enable_deduplication" in config:
            agent.enable_deduplication = bool(config["enable_deduplication"])
        
        if "similarity_threshold" in config:
            agent.similarity_threshold = max(0.0, min(1.0, float(config["similarity_threshold"])))
        
        if "default_strategy" in config:
            from ...agents.source_retrieval import RetrievalStrategy
            try:
                agent.default_strategy = RetrievalStrategy(config["default_strategy"])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid strategy: {config['default_strategy']}"
                )
        
        if "enable_multi_source" in config:
            agent.enable_multi_source = bool(config["enable_multi_source"])
        
        return {
            "agent_id": agent_id,
            "message": "Configuration updated successfully",
            "updated_configuration": {
                "max_results": agent.max_results,
                "min_relevance_threshold": agent.min_relevance_threshold,
                "semantic_weight": agent.semantic_weight,
                "keyword_weight": agent.keyword_weight,
                "enable_deduplication": agent.enable_deduplication,
                "similarity_threshold": agent.similarity_threshold,
                "default_strategy": agent.default_strategy.value,
                "enable_multi_source": agent.enable_multi_source
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
    "/strategies",
    summary="Get available retrieval strategies",
    description="List all available retrieval strategies and their descriptions"
)
async def get_retrieval_strategies() -> Dict[str, Any]:
    """
    Get information about available retrieval strategies.
    
    Returns:
        Dictionary of available strategies with descriptions
    """
    from ...agents.source_retrieval import RetrievalStrategy
    
    strategies = {
        RetrievalStrategy.SEMANTIC_ONLY.value: {
            "name": "Semantic Only",
            "description": "Uses only vector embeddings for semantic similarity search",
            "best_for": "Complex queries requiring conceptual understanding"
        },
        RetrievalStrategy.KEYWORD.value: {
            "name": "Keyword Search",
            "description": "Uses traditional keyword-based full-text search",
            "best_for": "Specific term lookups and exact matches"
        },
        RetrievalStrategy.HYBRID.value: {
            "name": "Hybrid Search",
            "description": "Combines semantic and keyword search for balanced results",
            "best_for": "Most queries requiring both precision and recall"
        },
        RetrievalStrategy.ADAPTIVE.value: {
            "name": "Adaptive Strategy",
            "description": "Automatically selects the best strategy based on query characteristics",
            "best_for": "General use when strategy selection should be automatic"
        }
    }
    
    return {
        "available_strategies": strategies,
        "default_strategy": RetrievalStrategy.ADAPTIVE.value,
        "recommendation": "Use 'adaptive' for automatic strategy selection or 'hybrid' for balanced results"
    } 
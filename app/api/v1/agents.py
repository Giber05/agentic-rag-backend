"""
API endpoints for agent framework management.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from ...core.dependencies import get_agent_registry, get_agent_coordinator, get_agent_metrics
from ...agents.registry import AgentRegistry
from ...agents.coordinator import AgentCoordinator, PipelineExecution
from ...agents.metrics import AgentMetrics
from ...models.agent_models import (
    AgentStateResponse,
    AgentRegistryStatsResponse,
    PipelineExecutionResponse,
    AgentMetricsResponse,
    SystemOverviewResponse,
    CreateAgentRequest,
    ExecutePipelineRequest,
    ErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


# Agent Registry Endpoints

@router.get("/registry/stats", response_model=AgentRegistryStatsResponse)
async def get_registry_stats(
    registry: AgentRegistry = Depends(get_agent_registry)
):
    """Get agent registry statistics."""
    try:
        stats = registry.get_registry_stats()
        return AgentRegistryStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting registry stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get registry stats: {str(e)}"
        )


@router.get("/registry/agents", response_model=List[AgentStateResponse])
async def list_agents(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    registry: AgentRegistry = Depends(get_agent_registry)
):
    """List all registered agents."""
    try:
        if agent_type:
            agents = registry.get_agents_by_type(agent_type)
        else:
            agents = registry.list_agents()
        
        return [
            AgentStateResponse(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type,
                status=agent.state.status,
                created_at=agent.state.created_at,
                started_at=agent.state.started_at,
                stopped_at=agent.state.stopped_at,
                last_activity=agent.state.last_activity,
                error_message=agent.state.error_message,
                metadata=agent.state.metadata,
                is_healthy=agent.is_healthy,
                is_running=agent.is_running
            )
            for agent in agents
        ]
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )


@router.get("/registry/agents/{agent_id}", response_model=AgentStateResponse)
async def get_agent_state(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry)
):
    """Get the state of a specific agent."""
    try:
        agent = registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        return AgentStateResponse(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            status=agent.state.status,
            created_at=agent.state.created_at,
            started_at=agent.state.started_at,
            stopped_at=agent.state.stopped_at,
            last_activity=agent.state.last_activity,
            error_message=agent.state.error_message,
            metadata=agent.state.metadata,
            is_healthy=agent.is_healthy,
            is_running=agent.is_running
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent state: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent state: {str(e)}"
        )


@router.post("/registry/agents", response_model=AgentStateResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: CreateAgentRequest,
    registry: AgentRegistry = Depends(get_agent_registry)
):
    """Create and register a new agent."""
    try:
        agent = await registry.create_agent(
            agent_type=request.agent_type,
            agent_id=request.agent_id,
            config=request.config,
            auto_start=request.auto_start
        )
        
        return AgentStateResponse(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            status=agent.state.status,
            created_at=agent.state.created_at,
            started_at=agent.state.started_at,
            stopped_at=agent.state.stopped_at,
            last_activity=agent.state.last_activity,
            error_message=agent.state.error_message,
            metadata=agent.state.metadata,
            is_healthy=agent.is_healthy,
            is_running=agent.is_running
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {str(e)}"
        )


@router.post("/registry/agents/{agent_id}/start")
async def start_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry)
):
    """Start a specific agent."""
    try:
        success = await registry.start_agent(agent_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to start agent {agent_id}"
            )
        
        return {"message": f"Agent {agent_id} started successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start agent: {str(e)}"
        )


@router.post("/registry/agents/{agent_id}/stop")
async def stop_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry)
):
    """Stop a specific agent."""
    try:
        success = await registry.stop_agent(agent_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to stop agent {agent_id}"
            )
        
        return {"message": f"Agent {agent_id} stopped successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop agent: {str(e)}"
        )


@router.delete("/registry/agents/{agent_id}")
async def unregister_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry)
):
    """Unregister an agent from the registry."""
    try:
        success = registry.unregister_agent(agent_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        return {"message": f"Agent {agent_id} unregistered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unregister agent: {str(e)}"
        )


# Pipeline Coordination Endpoints

@router.post("/pipeline/execute", response_model=PipelineExecutionResponse)
async def execute_pipeline(
    request: ExecutePipelineRequest,
    coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Execute the RAG pipeline for a query."""
    try:
        execution = await coordinator.execute_pipeline(
            query=request.query,
            conversation_id=request.conversation_id,
            context=request.context
        )
        
        return PipelineExecutionResponse(
            execution_id=execution.execution_id,
            query=execution.query,
            conversation_id=execution.conversation_id,
            status=execution.status,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            duration_ms=execution.duration_ms,
            current_step=execution.current_step,
            error_message=execution.error_message,
            metadata=execution.metadata,
            step_results={
                step: {
                    "agent_id": result.agent_id,
                    "agent_type": result.agent_type,
                    "success": result.success,
                    "processing_time_ms": result.processing_time_ms,
                    "data": result.data if result.success else None,
                    "error": result.error
                }
                for step, result in execution.step_results.items()
            }
        )
    except Exception as e:
        logger.error(f"Error executing pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute pipeline: {str(e)}"
        )


@router.get("/pipeline/executions", response_model=List[PipelineExecutionResponse])
async def list_pipeline_executions(
    active_only: bool = Query(False, description="Show only active executions"),
    limit: int = Query(50, description="Maximum number of executions to return", ge=1, le=1000),
    coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """List pipeline executions."""
    try:
        if active_only:
            executions = coordinator.list_active_executions()
        else:
            executions = coordinator.get_execution_history(limit)
        
        return [
            PipelineExecutionResponse(
                execution_id=execution.execution_id,
                query=execution.query,
                conversation_id=execution.conversation_id,
                status=execution.status,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                duration_ms=execution.duration_ms,
                current_step=execution.current_step,
                error_message=execution.error_message,
                metadata=execution.metadata,
                step_results={
                    step: {
                        "agent_id": result.agent_id,
                        "agent_type": result.agent_type,
                        "success": result.success,
                        "processing_time_ms": result.processing_time_ms,
                        "data": result.data if result.success else None,
                        "error": result.error
                    }
                    for step, result in execution.step_results.items()
                }
            )
            for execution in executions
        ]
    except Exception as e:
        logger.error(f"Error listing pipeline executions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list pipeline executions: {str(e)}"
        )


@router.get("/pipeline/executions/{execution_id}", response_model=PipelineExecutionResponse)
async def get_pipeline_execution(
    execution_id: str,
    coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Get details of a specific pipeline execution."""
    try:
        execution = coordinator.get_execution(execution_id)
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline execution {execution_id} not found"
            )
        
        return PipelineExecutionResponse(
            execution_id=execution.execution_id,
            query=execution.query,
            conversation_id=execution.conversation_id,
            status=execution.status,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            duration_ms=execution.duration_ms,
            current_step=execution.current_step,
            error_message=execution.error_message,
            metadata=execution.metadata,
            step_results={
                step: {
                    "agent_id": result.agent_id,
                    "agent_type": result.agent_type,
                    "success": result.success,
                    "processing_time_ms": result.processing_time_ms,
                    "data": result.data if result.success else None,
                    "error": result.error
                }
                for step, result in execution.step_results.items()
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline execution: {str(e)}"
        )


@router.post("/pipeline/executions/{execution_id}/cancel")
async def cancel_pipeline_execution(
    execution_id: str,
    coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Cancel an active pipeline execution."""
    try:
        success = await coordinator.cancel_execution(execution_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline execution {execution_id} not found or not active"
            )
        
        return {"message": f"Pipeline execution {execution_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling pipeline execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel pipeline execution: {str(e)}"
        )


# Metrics Endpoints

@router.get("/metrics/overview", response_model=SystemOverviewResponse)
async def get_system_overview(
    metrics: AgentMetrics = Depends(get_agent_metrics)
):
    """Get system-wide metrics overview."""
    try:
        overview = metrics.get_system_overview()
        return SystemOverviewResponse(**overview)
    except Exception as e:
        logger.error(f"Error getting system overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system overview: {str(e)}"
        )


@router.get("/metrics/agents/{agent_id}", response_model=AgentMetricsResponse)
async def get_agent_metrics(
    agent_id: str,
    metrics: AgentMetrics = Depends(get_agent_metrics)
):
    """Get performance metrics for a specific agent."""
    try:
        agent_metrics = metrics.get_agent_metrics(agent_id)
        if not agent_metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No metrics found for agent {agent_id}"
            )
        
        return AgentMetricsResponse(
            agent_id=agent_metrics.agent_id,
            agent_type=agent_metrics.agent_type,
            total_operations=agent_metrics.total_operations,
            successful_operations=agent_metrics.successful_operations,
            failed_operations=agent_metrics.failed_operations,
            success_rate=agent_metrics.success_rate,
            error_rate=agent_metrics.error_rate,
            average_processing_time_ms=agent_metrics.average_processing_time_ms,
            min_processing_time_ms=agent_metrics.min_processing_time_ms,
            max_processing_time_ms=agent_metrics.max_processing_time_ms,
            throughput_per_minute=agent_metrics.throughput_per_minute,
            last_operation_time=agent_metrics.last_operation_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent metrics: {str(e)}"
        )


@router.get("/metrics/anomalies")
async def get_anomalies(
    metrics: AgentMetrics = Depends(get_agent_metrics)
):
    """Get detected performance anomalies."""
    try:
        anomalies = metrics.detect_anomalies()
        return {"anomalies": anomalies}
    except Exception as e:
        logger.error(f"Error getting anomalies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get anomalies: {str(e)}"
        )


@router.get("/health")
async def agent_framework_health(
    registry: AgentRegistry = Depends(get_agent_registry),
    coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Get health status of the agent framework."""
    try:
        registry_stats = registry.get_registry_stats()
        coordinator_health = await coordinator.health_check()
        
        return {
            "agent_framework_healthy": True,
            "registry_stats": registry_stats,
            "coordinator_health": coordinator_health,
            "timestamp": "2024-01-01T00:00:00Z"  # Will be replaced with actual timestamp
        }
    except Exception as e:
        logger.error(f"Error checking agent framework health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check agent framework health: {str(e)}"
        ) 
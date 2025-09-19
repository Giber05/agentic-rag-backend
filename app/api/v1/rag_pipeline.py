"""
API endpoints for the RAG Pipeline Orchestrator.
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List
import logging
import json
import asyncio
import uuid
from datetime import datetime
from pydantic import BaseModel

from ...core.rag_pipeline import RAGPipelineOrchestrator, PipelineResult
from ...core.dependencies import get_agent_registry, get_agent_metrics, security_dependencies, authenticated_security_dependencies
from ...core.supabase_auth import get_current_user, get_optional_user
from ...models.rag_models import (
    RAGProcessRequest,
    RAGProcessResponse,
    RAGStreamRequest,
    PipelineStatusResponse,
    PipelineMetricsResponse,
    ErrorResponse,
    RAGRequest,
    ProcessingResult
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])

# Global pipeline orchestrator instance
_pipeline_orchestrator: Optional[RAGPipelineOrchestrator] = None
_optimized_orchestrator: Optional[Any] = None  # Import will be handled dynamically


def get_pipeline_orchestrator(
    agent_registry=Depends(get_agent_registry),
    agent_metrics=Depends(get_agent_metrics)
) -> RAGPipelineOrchestrator:
    """Get or create the RAG Pipeline Orchestrator."""
    global _pipeline_orchestrator
    
    if _pipeline_orchestrator is None:
        _pipeline_orchestrator = RAGPipelineOrchestrator(
            agent_registry=agent_registry,
            agent_metrics=agent_metrics,
            config={
                "max_pipeline_duration": 30.0,
                "enable_fallbacks": True,
                "enable_caching": True,
                "enable_streaming": True
            }
        )
    
    return _pipeline_orchestrator


def get_optimized_orchestrator(
    agent_registry=Depends(get_agent_registry),
    agent_metrics=Depends(get_agent_metrics)
) -> Any:
    """Get or create the Optimized RAG Pipeline Orchestrator."""
    global _optimized_orchestrator
    
    if _optimized_orchestrator is None:
        # Use the new token-tracking optimized orchestrator
        from ...core.rag_pipeline_optimized import OptimizedRAGPipelineOrchestrator
        _optimized_orchestrator = OptimizedRAGPipelineOrchestrator()
    
    return _optimized_orchestrator


@router.post(
    "/process",
    response_model=ProcessingResult,
    summary="Process RAG Query (Optimized)",
    description="Process a query through the optimized RAG pipeline for better cost efficiency."
)
async def process_rag_query(
    request: RAGRequest,
    use_full_pipeline: bool = Query(False, description="Use full pipeline instead of optimized version"),
    current_user = Depends(get_optional_user),  # Optional auth for public queries
    orchestrator: Any = Depends(get_optimized_orchestrator),
    full_orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> ProcessingResult:
    """
    Process a query through the RAG pipeline.
    
    By default uses the optimized pipeline for 80-90% cost savings.
    Set use_full_pipeline=true to use the complete pipeline.
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing RAG query {request_id}: '{request.query[:50]}...' (optimized: {not use_full_pipeline})")
        
        # Process the query
        if use_full_pipeline:
            # Use full orchestrator
            result = await full_orchestrator.process_query(
                query=request.query,
                source=request.source,
                conversation_history=request.conversation_history,
                user_context=request.user_context,
                pipeline_config=request.pipeline_config
            )
            
            # Convert PipelineResult to ProcessingResult format
            processing_result = ProcessingResult(
                request_id=result.request_id,
                query=request.query,
                source=request.source,
                status=result.status.value if hasattr(result.status, 'value') else str(result.status),
                pipeline_type="full",
                final_response=result.final_response or {},
                stage_results=result.stage_results or {},
                source_stats=result.stage_results.get("source_retrieval", {}).get("source_breakdown", {}),
                total_duration=result.total_duration,
                optimization_info={
                    "pipeline_used": "full",
                    "cost_saved": 0.0,  # Full pipeline doesn't save costs
                    "optimization": "none"
                }
            )
        else:
            # Use optimized orchestrator with token tracking
            processing_result = await orchestrator.process_query(request)
        
        logger.info(f"RAG query {request_id} completed in {processing_result.total_duration:.3f}s")
        return processing_result
        
    except Exception as e:
        error_msg = f"RAG pipeline processing failed: {str(e)}"
        logger.error(f"RAG query {request_id} failed: {error_msg}")
        
        return ProcessingResult(
            request_id=request_id,
            query=request.query,
            source=request.source,
            status="failed",
            pipeline_type="optimized" if not use_full_pipeline else "full",
            final_response={
                "query": request.query,
                "response": {
                    "content": f"I apologize, but I encountered an error while processing your query: {error_msg}",
                    "citations": [],
                    "format_type": "markdown"
                }
            },
            stage_results={"error": error_msg},
            source_stats={},
            total_duration=0.0,
            optimization_info={
                "pipeline_used": "optimized" if not use_full_pipeline else "full",
                "error": True
            }
        )


@router.post(
    "/process/full",
    response_model=ProcessingResult,
    summary="Process RAG Query (Full Pipeline)",
    description="Process a query through the complete RAG pipeline with all agents."
)
async def process_rag_query_full(
    request: RAGRequest,
    current_user = Depends(get_current_user),  # Require auth for full pipeline
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> ProcessingResult:
    """
    Process a query through the complete RAG pipeline.
    
    This endpoint uses all 4 agents for maximum accuracy but higher cost.
    For cost-efficient processing, use the /process endpoint instead.
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing RAG query {request_id} (FULL PIPELINE): '{request.query[:50]}...'")
        
        # Process the query through full pipeline
        result = await orchestrator.process_query(
            query=request.query,
            source=request.source,
            conversation_history=request.conversation_history,
            user_context=request.user_context,
            pipeline_config=request.pipeline_config
        )
        
        # Convert PipelineResult to ProcessingResult format
        processing_result = ProcessingResult(
            request_id=result.request_id,
            query=request.query,
            source=request.source,
            status=result.status.value if hasattr(result.status, 'value') else str(result.status),
            pipeline_type="full",
            final_response=result.final_response or {},
            stage_results=result.stage_results or {},
            source_stats=result.stage_results.get("source_retrieval", {}).get("source_breakdown", {}),
            total_duration=result.total_duration,
            optimization_info={
                "pipeline_used": "full",
                "cost_saved": 0.0,
                "optimization": "none"
            }
        )
        
        logger.info(f"Full RAG query {request_id} completed in {processing_result.total_duration:.3f}s")
        return processing_result
        
    except Exception as e:
        error_msg = f"Full RAG pipeline processing failed: {str(e)}"
        logger.error(f"Full RAG query {request_id} failed: {error_msg}")
        
        return ProcessingResult(
            request_id=request_id,
            query=request.query,
            source=request.source,
            status="failed",
            pipeline_type="full",
            final_response={
                "query": request.query,
                "response": {
                    "content": f"I apologize, but I encountered an error while processing your query: {error_msg}",
                    "citations": [],
                    "format_type": "markdown"
                }
            },
            stage_results={"error": error_msg},
            source_stats={},
            total_duration=0.0,
            optimization_info={
                "pipeline_used": "full",
                "error": True
            }
        )


@router.post(
    "/stream",
    summary="Stream query processing through RAG pipeline",
    description="Process a query through the RAG pipeline with real-time streaming updates."
)
async def stream_query(
    request: RAGStreamRequest,
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """Stream query processing through the RAG pipeline."""
    
    async def generate_stream():
        try:
            logger.info(f"Starting RAG streaming for query: '{request.query[:50]}...'")
            
            async for update in orchestrator.stream_query(
                query=request.query,
                conversation_history=request.conversation_history,
                user_context=request.user_context,
                pipeline_config=request.pipeline_config
            ):
                # Format as Server-Sent Events
                yield f"data: {json.dumps(update)}\n\n"
                
        except Exception as e:
            logger.error(f"RAG streaming failed: {str(e)}")
            error_update = {
                "stage": "failed",
                "status": "error",
                "error": str(e),
                "timestamp": "2024-01-01T00:00:00Z"
            }
            yield f"data: {json.dumps(error_update)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@router.websocket("/stream-ws")
async def websocket_stream_query(
    websocket: WebSocket,
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """WebSocket endpoint for real-time RAG pipeline streaming."""
    await websocket.accept()
    
    try:
        # Receive initial request
        request_data = await websocket.receive_text()
        request = json.loads(request_data)
        
        query = request.get("query", "")
        conversation_history = request.get("conversation_history", [])
        user_context = request.get("user_context", {})
        pipeline_config = request.get("pipeline_config", {})
        
        logger.info(f"WebSocket RAG streaming for query: '{query[:50]}...'")
        
        # Stream pipeline updates
        async for update in orchestrator.stream_query(
            query=query,
            conversation_history=conversation_history,
            user_context=user_context,
            pipeline_config=pipeline_config
        ):
            await websocket.send_text(json.dumps(update))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket RAG streaming failed: {str(e)}")
        error_update = {
            "stage": "failed",
            "status": "error",
            "error": str(e)
        }
        try:
            await websocket.send_text(json.dumps(error_update))
        except:
            pass  # Client may have disconnected
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.options(
    "/pipeline/status",
    summary="CORS preflight for pipeline status",
    description="Handle CORS preflight requests for pipeline status endpoint."
)
async def options_pipeline_status():
    """Handle OPTIONS request for CORS preflight."""
    return {"message": "OK"}


@router.get(
    "/pipeline/status",
    response_model=PipelineStatusResponse,
    summary="Get pipeline status",
    description="Get current status and metrics of the RAG pipeline."
)
async def get_pipeline_status(
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> PipelineStatusResponse:
    """Get current pipeline status and metrics."""
    try:
        status = orchestrator.get_pipeline_status()
        
        return PipelineStatusResponse(
            active_pipelines=status["active_pipelines"],
            cached_results=status["cached_results"],
            statistics=status["statistics"],
            configuration=status["configuration"],
            timestamp=status["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")


@router.get(
    "/pipeline/metrics",
    response_model=PipelineMetricsResponse,
    summary="Get pipeline performance metrics",
    description="Get detailed performance metrics for the RAG pipeline."
)
async def get_pipeline_metrics(
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> PipelineMetricsResponse:
    """Get pipeline performance metrics."""
    try:
        status = orchestrator.get_pipeline_status()
        active_pipelines = orchestrator.get_active_pipelines()
        
        return PipelineMetricsResponse(
            total_pipelines=status["statistics"]["total_pipelines"],
            successful_pipelines=status["statistics"]["successful_pipelines"],
            failed_pipelines=status["statistics"]["failed_pipelines"],
            success_rate=(
                status["statistics"]["successful_pipelines"] / 
                max(1, status["statistics"]["total_pipelines"])
            ),
            avg_duration=status["statistics"]["avg_duration"],
            stage_performance=status["statistics"]["stage_performance"],
            active_pipelines=active_pipelines,
            cache_stats={
                "cached_results": status["cached_results"],
                "cache_hit_rate": 0.0  # TODO: Implement cache hit rate tracking
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get pipeline metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline metrics: {str(e)}")


@router.get(
    "/pipeline/active",
    summary="Get active pipelines",
    description="Get information about currently active pipeline executions."
)
async def get_active_pipelines(
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> Dict[str, Any]:
    """Get information about currently active pipelines."""
    try:
        active_pipelines = orchestrator.get_active_pipelines()
        
        return {
            "active_pipelines": active_pipelines,
            "count": len(active_pipelines),
            "timestamp": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Failed to get active pipelines: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active pipelines: {str(e)}")


@router.post(
    "/pipeline/configure",
    summary="Configure pipeline settings",
    description="Update RAG pipeline configuration settings."
)
async def configure_pipeline(
    config: Dict[str, Any],
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> Dict[str, Any]:
    """Configure pipeline settings."""
    try:
        # Update orchestrator configuration
        for key, value in config.items():
            if hasattr(orchestrator, key):
                setattr(orchestrator, key, value)
                logger.info(f"Updated pipeline config: {key} = {value}")
        
        # Update agent configurations
        if "agent_configs" in config:
            orchestrator.agent_configs.update(config["agent_configs"])
            logger.info("Updated agent configurations")
        
        return {
            "message": "Pipeline configuration updated successfully",
            "updated_config": config,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Failed to configure pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure pipeline: {str(e)}")


@router.delete(
    "/pipeline/cache",
    summary="Clear pipeline cache",
    description="Clear the RAG pipeline result cache."
)
async def clear_pipeline_cache(
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> Dict[str, Any]:
    """Clear pipeline cache."""
    try:
        cache_count = len(orchestrator.pipeline_cache)
        orchestrator.pipeline_cache.clear()
        
        logger.info(f"Cleared {cache_count} cached pipeline results")
        
        return {
            "message": f"Cleared {cache_count} cached results",
            "cache_count": cache_count,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear pipeline cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear pipeline cache: {str(e)}")


@router.get(
    "/health",
    summary="Pipeline health check",
    description="Check the health status of the RAG pipeline and all agents."
)
async def pipeline_health_check(
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> Dict[str, Any]:
    """Check pipeline health status."""
    try:
        # Check agent registry
        agent_types = ["query_rewriter", "context_decision", "source_retrieval", "answer_generation"]
        agent_health = {}
        
        for agent_type in agent_types:
            agents = orchestrator.agent_registry.get_agents_by_type(agent_type)
            if agents:
                agent = agents[0]
                agent_health[agent_type] = {
                    "available": True,
                    "running": agent.is_running,
                    "healthy": agent.is_healthy,
                    "agent_id": agent.agent_id
                }
            else:
                agent_health[agent_type] = {
                    "available": False,
                    "running": False,
                    "healthy": False,
                    "agent_id": None
                }
        
        # Overall health status
        all_healthy = all(
            health["available"] and health["running"] and health["healthy"]
            for health in agent_health.values()
        )
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "pipeline_orchestrator": {
                "active_pipelines": len(orchestrator.active_pipelines),
                "cached_results": len(orchestrator.pipeline_cache),
                "total_processed": orchestrator.pipeline_stats["total_pipelines"]
            },
            "agents": agent_health,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Pipeline health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        } 


# Intelligent MCP Endpoints

@router.post(
    "/intelligent/preview",
    summary="Preview intelligent MCP operations",
    description="Preview the operation plan for a query without executing it."
)
async def preview_intelligent_operations(
    request: Dict[str, Any],
    current_user = Depends(get_optional_user)
) -> Dict[str, Any]:
    """Preview intelligent MCP operation plan without execution."""
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing required parameter: query")
        
        # Import the intelligent MCP agent
        from ...agents.intelligent_mcp_agent import IntelligentMCPAgent
        
        # Create agent instance
        agent = IntelligentMCPAgent()
        await agent.initialize()
        
        # Get operation preview
        preview = await agent.get_operation_preview(
            query=query,
            context=request.get("context", {})
        )
        
        return {
            "query": query,
            "preview": preview,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to preview intelligent operations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to preview operations: {str(e)}")


@router.get(
    "/intelligent/health",
    summary="Intelligent MCP agent health check",
    description="Check the health status of the intelligent MCP agent and its dependencies."
)
async def intelligent_mcp_health_check() -> Dict[str, Any]:
    """Check intelligent MCP agent health status."""
    try:
        # Import the intelligent MCP agent
        from ...agents.intelligent_mcp_agent import IntelligentMCPAgent
        
        # Create and check agent health
        agent = IntelligentMCPAgent()
        await agent.initialize()
        
        health_check = await agent.health_check()
        
        return {
            "agent_health": health_check,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Intelligent MCP health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get(
    "/intelligent/statistics",
    summary="Get intelligent MCP agent statistics",
    description="Get performance statistics and metrics for the intelligent MCP agent."
)
async def get_intelligent_mcp_statistics() -> Dict[str, Any]:
    """Get intelligent MCP agent statistics."""
    try:
        # Import the intelligent MCP agent
        from ...agents.intelligent_mcp_agent import IntelligentMCPAgent
        
        # Create agent instance
        agent = IntelligentMCPAgent()
        await agent.initialize()
        
        statistics = agent.get_agent_statistics()
        
        return {
            "statistics": statistics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get intelligent MCP statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post(
    "/intelligent/configure",
    summary="Configure intelligent MCP agent",
    description="Update configuration settings for the intelligent MCP agent."
)
async def configure_intelligent_mcp(
    config: Dict[str, Any],
    current_user = Depends(get_current_user)  # Require auth for configuration changes
) -> Dict[str, Any]:
    """Configure intelligent MCP agent settings."""
    try:
        # Import the intelligent MCP agent
        from ...agents.intelligent_mcp_agent import IntelligentMCPAgent
        
        # Create agent instance
        agent = IntelligentMCPAgent()
        await agent.initialize()
        
        # Update configuration
        result = agent.update_config(config)
        
        return {
            "message": "Intelligent MCP agent configuration updated successfully",
            "updated_config": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to configure intelligent MCP agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure agent: {str(e)}")


# ================================
# PHASE 4: Configuration Management Endpoints
# ================================

class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates."""
    updates: Dict[str, Any]


class TemplateRequest(BaseModel):
    """Request model for operation template management."""
    name: str
    intent_types: List[str]
    operations: List[Dict[str, Any]]
    conditions: Optional[Dict[str, Any]] = {}
    priority: int = 1
    enabled: bool = True
    description: str = ""


@router.get(
    "/config",
    summary="Get intelligent MCP configuration",
    description="Retrieve the current intelligent MCP configuration including all settings and templates."
)
async def get_intelligent_mcp_config(
    current_user = Depends(get_optional_user)
) -> Dict[str, Any]:
    """Get the current intelligent MCP configuration."""
    try:
        from ...core.intelligent_config import get_intelligent_config
        config = get_intelligent_config()
        
        return {
            "success": True,
            "config": config.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get intelligent MCP config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve configuration: {str(e)}"
        )


@router.put(
    "/config",
    summary="Update intelligent MCP configuration",
    description="Update intelligent MCP configuration settings. Requires authentication."
)
async def update_intelligent_mcp_config(
    request: ConfigUpdateRequest,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update intelligent MCP configuration settings."""
    try:
        from ...core.intelligent_config import update_intelligent_config, get_config_manager
        
        # Update configuration
        updated_config = update_intelligent_config(request.updates)
        
        # Save to file
        config_manager = get_config_manager()
        save_success = config_manager.save_config(updated_config)
        
        return {
            "success": True,
            "config": updated_config.dict(),
            "saved_to_file": save_success,
            "updated_fields": list(request.updates.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update intelligent MCP config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration: {str(e)}"
        )


@router.get(
    "/config/templates",
    summary="Get operation templates",
    description="Retrieve all available operation templates for intelligent MCP operations."
)
async def get_operation_templates(
    current_user = Depends(get_optional_user)
) -> Dict[str, Any]:
    """Get all operation templates."""
    try:
        from ...core.intelligent_config import get_config_manager
        
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        templates = []
        for template in config.templates:
            templates.append(template.dict())
        
        return {
            "success": True,
            "templates": templates,
            "total_templates": len(templates),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get operation templates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve templates: {str(e)}"
        )


@router.post(
    "/config/templates",
    summary="Add operation template",
    description="Add a new operation template for intelligent MCP operations. Requires authentication."
)
async def add_operation_template(
    template_request: TemplateRequest,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Add a new operation template."""
    try:
        from ...core.intelligent_config import get_config_manager, OperationTemplate
        
        config_manager = get_config_manager()
        
        # Create new template
        new_template = OperationTemplate(
            name=template_request.name,
            intent_types=template_request.intent_types,
            operations=template_request.operations,
            conditions=template_request.conditions,
            priority=template_request.priority,
            enabled=template_request.enabled,
            description=template_request.description
        )
        
        # Add template
        success = config_manager.add_template(new_template)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Template '{template_request.name}' already exists"
            )
        
        # Save configuration
        save_success = config_manager.save_config()
        
        return {
            "success": True,
            "template": new_template.dict(),
            "saved_to_file": save_success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add operation template: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add template: {str(e)}"
        )


@router.delete(
    "/config/templates/{template_name}",
    summary="Remove operation template",
    description="Remove an operation template by name. Requires authentication."
)
async def remove_operation_template(
    template_name: str,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Remove an operation template by name."""
    try:
        from ...core.intelligent_config import get_config_manager
        
        config_manager = get_config_manager()
        
        # Remove template
        success = config_manager.remove_template(template_name)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Template '{template_name}' not found"
            )
        
        # Save configuration
        save_success = config_manager.save_config()
        
        return {
            "success": True,
            "removed_template": template_name,
            "saved_to_file": save_success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove operation template: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove template: {str(e)}"
        )


@router.get(
    "/config/performance",
    summary="Get performance settings",
    description="Retrieve current performance and resource configuration settings."
)
async def get_performance_config(
    current_user = Depends(get_optional_user)
) -> Dict[str, Any]:
    """Get performance configuration settings."""
    try:
        from ...core.intelligent_config import get_intelligent_config
        
        config = get_intelligent_config()
        
        return {
            "success": True,
            "performance": config.performance.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve performance configuration: {str(e)}"
        )


@router.put(
    "/config/performance",
    summary="Update performance settings",
    description="Update performance and resource configuration settings. Requires authentication."
)
async def update_performance_config(
    performance_updates: Dict[str, Any],
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update performance configuration settings."""
    try:
        from ...core.intelligent_config import update_intelligent_config
        
        # Wrap performance updates in the correct structure
        updates = {"performance": performance_updates}
        
        # Update configuration
        updated_config = update_intelligent_config(updates)
        
        return {
            "success": True,
            "performance": updated_config.performance.dict(),
            "updated_fields": list(performance_updates.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update performance config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update performance configuration: {str(e)}"
        )


@router.post(
    "/config/reset",
    summary="Reset configuration to defaults",
    description="Reset intelligent MCP configuration to default values. Requires authentication."
)
async def reset_intelligent_mcp_config(
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Reset configuration to default values."""
    try:
        from ...core.intelligent_config import get_config_manager, IntelligentMCPConfig
        
        config_manager = get_config_manager()
        
        # Create default configuration
        default_config = IntelligentMCPConfig()
        
        # Save default configuration
        save_success = config_manager.save_config(default_config)
        
        return {
            "success": True,
            "config": default_config.dict(),
            "saved_to_file": save_success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset intelligent MCP config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset configuration: {str(e)}"
        ) 
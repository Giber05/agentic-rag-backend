"""
Agent coordinator for orchestrating the RAG pipeline.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from uuid import uuid4

from .base import BaseAgent, AgentResult, AgentMessage, AgentStatus
from .registry import AgentRegistry

logger = logging.getLogger(__name__)


class PipelineStep:
    """Represents a step in the RAG pipeline."""
    
    def __init__(
        self,
        step_name: str,
        agent_type: str,
        required: bool = True,
        timeout_seconds: float = 30.0,
        retry_count: int = 2
    ):
        self.step_name = step_name
        self.agent_type = agent_type
        self.required = required
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count


class PipelineExecution:
    """Tracks the execution of a pipeline."""
    
    def __init__(self, execution_id: str, query: str, conversation_id: Optional[str] = None):
        self.execution_id = execution_id
        self.query = query
        self.conversation_id = conversation_id
        self.started_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.status = "running"
        self.current_step: Optional[str] = None
        self.step_results: Dict[str, AgentResult] = {}
        self.error_message: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    @property
    def duration_ms(self) -> int:
        """Get execution duration in milliseconds."""
        end_time = self.completed_at or datetime.utcnow()
        return int((end_time - self.started_at).total_seconds() * 1000)


class AgentCoordinator:
    """
    Coordinates the execution of the RAG pipeline through multiple agents.
    """
    
    def __init__(self, agent_registry: AgentRegistry):
        """Initialize the agent coordinator."""
        self.agent_registry = agent_registry
        self._pipeline_steps: List[PipelineStep] = []
        self._active_executions: Dict[str, PipelineExecution] = {}
        self._execution_history: List[PipelineExecution] = []
        self._max_history_size = 1000
        
        # Default RAG pipeline configuration
        self._setup_default_pipeline()
        
        logger.info("Agent coordinator initialized")
    
    def _setup_default_pipeline(self) -> None:
        """Set up the default RAG pipeline steps."""
        self._pipeline_steps = [
            PipelineStep("query_rewriting", "query_rewriter", required=False, timeout_seconds=10.0),
            PipelineStep("context_decision", "context_decision", required=True, timeout_seconds=5.0),
            PipelineStep("source_retrieval", "source_retrieval", required=True, timeout_seconds=15.0),
            PipelineStep("answer_generation", "answer_generation", required=True, timeout_seconds=30.0),
            PipelineStep("validation_refinement", "validation_refinement", required=False, timeout_seconds=20.0)
        ]
        
        logger.info(f"Configured default pipeline with {len(self._pipeline_steps)} steps")
    
    async def execute_pipeline(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        stream_callback: Optional[callable] = None
    ) -> PipelineExecution:
        """Execute the RAG pipeline for a given query."""
        execution_id = str(uuid4())
        execution = PipelineExecution(execution_id, query, conversation_id)
        execution.metadata.update(context or {})
        
        self._active_executions[execution_id] = execution
        
        try:
            logger.info(f"Starting pipeline execution {execution_id} for query: {query[:100]}...")
            
            # Initialize pipeline data
            pipeline_data = {
                "query": query,
                "conversation_id": conversation_id,
                "execution_id": execution_id,
                "context": context or {},
                "step_results": {}
            }
            
            # Execute each pipeline step
            for step in self._pipeline_steps:
                try:
                    execution.current_step = step.step_name
                    
                    # Notify stream callback if provided
                    if stream_callback:
                        await stream_callback({
                            "type": "step_started",
                            "execution_id": execution_id,
                            "step": step.step_name,
                            "agent_type": step.agent_type
                        })
                    
                    logger.info(f"Executing step: {step.step_name} with agent type: {step.agent_type}")
                    
                    # Execute the step
                    step_result = await self._execute_step(step, pipeline_data)
                    execution.step_results[step.step_name] = step_result
                    
                    # Update pipeline data with step result
                    pipeline_data["step_results"][step.step_name] = step_result.data
                    
                    # Check if step was successful
                    if not step_result.success:
                        if step.required:
                            raise RuntimeError(f"Required step {step.step_name} failed: {step_result.error}")
                        else:
                            logger.warning(f"Optional step {step.step_name} failed: {step_result.error}")
                    
                    # Notify stream callback of step completion
                    if stream_callback:
                        await stream_callback({
                            "type": "step_completed",
                            "execution_id": execution_id,
                            "step": step.step_name,
                            "success": step_result.success,
                            "processing_time_ms": step_result.processing_time_ms,
                            "data": step_result.data if step_result.success else None,
                            "error": step_result.error
                        })
                    
                except Exception as e:
                    logger.error(f"Error in pipeline step {step.step_name}: {str(e)}")
                    
                    if step.required:
                        execution.status = "failed"
                        execution.error_message = f"Step {step.step_name} failed: {str(e)}"
                        raise
                    else:
                        logger.warning(f"Optional step {step.step_name} failed, continuing pipeline")
            
            # Pipeline completed successfully
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            
            logger.info(f"Pipeline execution {execution_id} completed in {execution.duration_ms}ms")
            
        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            
            logger.error(f"Pipeline execution {execution_id} failed: {str(e)}")
        
        finally:
            # Move to history and clean up
            self._move_to_history(execution_id)
        
        return execution
    
    async def _execute_step(self, step: PipelineStep, pipeline_data: Dict[str, Any]) -> AgentResult:
        """Execute a single pipeline step."""
        # Get agents of the required type
        agents = self.agent_registry.get_agents_by_type(step.agent_type)
        
        if not agents:
            raise RuntimeError(f"No agents of type {step.agent_type} available")
        
        # Use the first available healthy agent
        agent = None
        for candidate in agents:
            if candidate.is_healthy:
                agent = candidate
                break
        
        if not agent:
            raise RuntimeError(f"No healthy agents of type {step.agent_type} available")
        
        # Prepare input data for the agent
        input_data = {
            "step_name": step.step_name,
            "pipeline_data": pipeline_data,
            "query": pipeline_data["query"],
            "conversation_id": pipeline_data.get("conversation_id"),
            "context": pipeline_data.get("context", {}),
            "previous_results": pipeline_data.get("step_results", {})
        }
        
        # Execute with timeout and retry logic
        for attempt in range(step.retry_count + 1):
            try:
                logger.debug(f"Executing {step.step_name} (attempt {attempt + 1})")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    agent.process(input_data),
                    timeout=step.timeout_seconds
                )
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Step {step.step_name} timed out (attempt {attempt + 1})")
                if attempt == step.retry_count:
                    raise RuntimeError(f"Step {step.step_name} timed out after {step.retry_count + 1} attempts")
                
            except Exception as e:
                logger.warning(f"Step {step.step_name} failed (attempt {attempt + 1}): {str(e)}")
                if attempt == step.retry_count:
                    raise
                
                # Wait before retry
                await asyncio.sleep(1.0 * (attempt + 1))
        
        # Should not reach here
        raise RuntimeError(f"Step {step.step_name} failed after all retry attempts")
    
    def _move_to_history(self, execution_id: str) -> None:
        """Move an execution from active to history."""
        if execution_id in self._active_executions:
            execution = self._active_executions.pop(execution_id)
            self._execution_history.append(execution)
            
            # Trim history if it gets too large
            if len(self._execution_history) > self._max_history_size:
                self._execution_history = self._execution_history[-self._max_history_size:]
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """
        Get coordinator statistics.
        
        Returns:
            Dictionary containing coordinator statistics
        """
        completed_executions = [e for e in self._execution_history if e.status == "completed"]
        failed_executions = [e for e in self._execution_history if e.status == "failed"]
        
        total_executions = len(self._execution_history)
        avg_duration = 0.0
        if completed_executions:
            avg_duration = sum(e.duration_ms for e in completed_executions) / len(completed_executions)
        
        return {
            "total_executions": total_executions,
            "active_executions": len(self._active_executions),
            "completed_executions": len(completed_executions),
            "failed_executions": len(failed_executions),
            "success_rate": len(completed_executions) / total_executions if total_executions > 0 else 0.0,
            "average_duration_ms": avg_duration,
            "pipeline_steps": len(self._pipeline_steps),
            "configured_steps": [step.step_name for step in self._pipeline_steps]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the coordinator and its agents.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            # Check agent registry health
            registry_stats = self.agent_registry.get_registry_stats()
            healthy_agents = registry_stats.get("healthy_agents", 0)
            total_agents = registry_stats.get("total_agents", 0)
            
            # Check if we have agents for each required pipeline step
            missing_agent_types = []
            for step in self._pipeline_steps:
                if step.required:
                    agents = self.agent_registry.get_agents_by_type(step.agent_type)
                    healthy_agents_for_step = [a for a in agents if a.is_healthy]
                    if not healthy_agents_for_step:
                        missing_agent_types.append(step.agent_type)
            
            coordinator_healthy = len(missing_agent_types) == 0
            
            return {
                "coordinator_healthy": coordinator_healthy,
                "total_agents": total_agents,
                "healthy_agents": healthy_agents,
                "missing_required_agent_types": missing_agent_types,
                "pipeline_steps_configured": len(self._pipeline_steps),
                "active_executions": len(self._active_executions),
                "registry_healthy": healthy_agents == total_agents if total_agents > 0 else True
            }
            
        except Exception as e:
            logger.error(f"Error during health check: {str(e)}")
            return {
                "coordinator_healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_active_executions(self) -> List[PipelineExecution]:
        """Get list of currently active executions."""
        return list(self._active_executions.values())
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[PipelineExecution]:
        """
        Get execution history.
        
        Args:
            limit: Maximum number of executions to return
            
        Returns:
            List of pipeline executions
        """
        history = self._execution_history
        if limit:
            history = history[-limit:]
        return history
    
    def get_execution(self, execution_id: str) -> Optional[PipelineExecution]:
        """
        Get a specific execution by ID.
        
        Args:
            execution_id: ID of the execution to retrieve
            
        Returns:
            Pipeline execution or None if not found
        """
        # Check active executions first
        if execution_id in self._active_executions:
            return self._active_executions[execution_id]
        
        # Check history
        for execution in self._execution_history:
            if execution.execution_id == execution_id:
                return execution
        
        return None 
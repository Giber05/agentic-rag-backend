"""
Intelligent MCP Agent - AI Agent for Smart Operation Chaining

This agent integrates the MCP Operation Orchestrator into the agent framework,
providing intelligent chained operations across Jira and Confluence.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator

from .base import BaseAgent, AgentStatus
from ..services.mcp_operation_orchestrator import (
    MCPOperationOrchestrator,
    ChainResult,
    OperationPlan,
    QueryIntent
)
from ..services.mcp_atlassian_service import MCPAtlassianService
from ..utils.performance_monitor import get_performance_monitor, monitor_operation

logger = logging.getLogger(__name__)


class IntelligentMCPAgent(BaseAgent):
    """
    Intelligent MCP Agent that uses orchestrated chaining of operations.
    
    This agent analyzes user queries and executes intelligent operation chains
    across Jira and Confluence, similar to Cursor's assistant capabilities.
    """
    
    def __init__(
        self, 
        agent_id: str = "intelligent_mcp",
        config: Optional[Dict[str, Any]] = None
    ):
        # Load configuration from config system
        from ..core.intelligent_config import get_intelligent_config
        self.global_config = get_intelligent_config()
        
        # Default configuration from global config
        default_config = {
            "max_operations": self.global_config.performance.max_operations,
            "timeout": self.global_config.performance.timeout,
            "enable_streaming": self.global_config.enable_streaming,
            "enable_caching": self.global_config.performance.enable_caching,
            "confidence_threshold": self.global_config.performance.confidence_threshold,
            "cost_limit": self.global_config.performance.cost_limit
        }
        
        # Merge with provided config
        effective_config = {**default_config, **(config or {})}
        
        super().__init__(agent_id, "IntelligentMCPAgent", effective_config)
        
        # Initialize services
        self.mcp_service = MCPAtlassianService()
        self.orchestrator = MCPOperationOrchestrator(self.mcp_service)
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "successful_chains": 0,
            "failed_chains": 0,
            "avg_chain_duration": 0.0,
            "avg_confidence": 0.0,
            "intent_distribution": {},
            "cache_hits": 0
        }
        
        # Simple caching mechanism (TTL from config)
        self._cache = {}
        self._cache_ttl = self.global_config.performance.cache_ttl
    
    async def initialize(self) -> bool:
        """Initialize the agent and its dependencies."""
        try:
            logger.info(f"Initializing {self.agent_type} agent: {self.agent_id}")
            
            # Initialize orchestrator
            if not await self.orchestrator.initialize():
                logger.error("Failed to initialize orchestrator")
                return False
            
            self.state.status = AgentStatus.IDLE
            logger.info(f"Intelligent MCP Agent {self.agent_id} initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Intelligent MCP Agent: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            return False
    
    async def _process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input using intelligent operation chaining.
        
        Args:
            input_data: Dictionary containing:
                - query: User query to process
                - context: Optional context data
                - config: Optional configuration overrides
                
        Returns:
            Dictionary containing chain results and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Extract required parameters
            query = input_data.get("query")
            if not query:
                raise ValueError("Missing required parameter: query")
            
            context = input_data.get("context", {})
            config_overrides = input_data.get("config", {})
            
            # Apply configuration overrides
            effective_config = {**self.config, **config_overrides}
            
            logger.info(f"Processing query with intelligent chaining: '{query}'")
            
            # Check cache first
            cache_key = self._generate_cache_key(query, context)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                self.stats["cache_hits"] += 1
                return cached_result
            
            # Step 1: Pre-validate query
            validation_result = self._validate_query(query, effective_config)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["reason"],
                    "query": query,
                    "processing_time": (datetime.utcnow() - start_time).total_seconds()
                }
            
            # Step 2: Execute intelligent chaining with performance monitoring
            performance_monitor = get_performance_monitor()
            async with performance_monitor.monitor_operation("intelligent_chain_execution", {"query_length": len(query)}) as metrics:
                # Use adaptive AI-driven execution
                chain_result = await self.orchestrator.adaptive_execute(
                    query=query,
                    context=context,
                    max_steps=effective_config["max_operations"],
                    timeout=effective_config["timeout"]
                )
                
                # Add chain metadata to metrics
                metrics["metadata"] = {
                    "query_length": len(query),
                    "operations_executed": len(chain_result.results),
                    "intent": chain_result.intent.value,
                    "confidence": chain_result.metadata.get("plan_confidence", 0.0)
                }
            
            # Step 3: Post-process results
            processed_result = self._post_process_chain_result(chain_result, effective_config)
            
            # Step 4: Cache successful results
            if chain_result.success and effective_config["enable_caching"]:
                self._store_in_cache(cache_key, processed_result)
            
            # Step 5: Update statistics
            self._update_agent_stats(chain_result, start_time)
            
            logger.info(f"Chain processing completed in {chain_result.total_duration:.2f}s")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.stats["failed_chains"] += 1
            
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "processing_time": processing_time,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def stream_process(
        self,
        input_data: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream processing results as operations complete.
        
        Args:
            input_data: Same as _process method
            
        Yields:
            Progressive updates as operations complete
        """
        try:
            query = input_data.get("query")
            context = input_data.get("context", {})
            config_overrides = input_data.get("config", {})
            
            # Apply configuration
            effective_config = {**self.config, **config_overrides}
            
            if not effective_config["enable_streaming"]:
                # Fall back to non-streaming
                result = await self._process(input_data)
                yield result
                return
            
            logger.info(f"Starting streaming process for query: '{query}'")
            
            # Yield initial status
            yield {
                "status": "started",
                "query": query,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Step 1: Analyze query and create plan
            try:
                plan = await self.orchestrator.analyze_query(
                    query,
                    context,
                    effective_config["max_operations"]
                )
                
                # Yield plan information
                yield {
                    "status": "plan_created",
                    "plan": {
                        "intent": plan.intent.value,
                        "steps": len(plan.steps),
                        "estimated_duration": plan.estimated_duration,
                        "confidence": plan.confidence,
                        "reasoning": plan.reasoning
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                yield {
                    "status": "plan_failed",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                return
            
            # Step 2: Execute operations with streaming updates
            try:
                # For streaming, we'll need to modify the orchestrator or implement here
                # For now, execute the plan and yield step-by-step updates
                
                chain_result = await self.orchestrator.adaptive_execute(
                    query,
                    context,
                    max_steps=effective_config["max_operations"],
                    timeout=effective_config["timeout"]
                )
                
                # Yield step results
                for i, step_result in enumerate(chain_result.results):
                    yield {
                        "status": "step_completed",
                        "step": i + 1,
                        "total_steps": len(chain_result.results),
                        "step_id": step_result.step_id,
                        "operation": step_result.operation_type.value,
                        "success": step_result.success,
                        "duration": step_result.duration,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # Yield final result
                processed_result = self._post_process_chain_result(chain_result, effective_config)
                yield {
                    "status": "completed",
                    **processed_result,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                yield {
                    "status": "execution_failed",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}")
            yield {
                "status": "stream_error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_operation_preview(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a preview of operations that would be executed without running them.
        
        Args:
            query: User query to analyze
            context: Optional context data
            
        Returns:
            Dictionary with operation plan preview
        """
        try:
            logger.info(f"Generating operation preview for: '{query}'")
            
            plan = await self.orchestrator.analyze_query(
                query,
                context or {},
                self.config["max_operations"]
            )
            
            return {
                "success": True,
                "query": query,
                "plan": {
                    "plan_id": plan.plan_id,
                    "intent": plan.intent.value,
                    "confidence": plan.confidence,
                    "reasoning": plan.reasoning,
                    "estimated_cost": plan.estimated_cost,
                    "estimated_duration": plan.estimated_duration,
                    "operations": [
                        {
                            "step_id": step.step_id,
                            "operation_type": step.operation_type.value,
                            "description": step.description,
                            "parameters": step.parameters,
                            "depends_on": step.depends_on,
                            "condition": step.condition
                        }
                        for step in plan.steps
                    ]
                },
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating operation preview: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update agent configuration.
        
        Args:
            new_config: Configuration updates
            
        Returns:
            Updated configuration
        """
        self.config.update(new_config)
        logger.info(f"Updated agent configuration: {new_config}")
        return self.config
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get detailed agent statistics."""
        performance_monitor = get_performance_monitor()
        
        return {
            "agent_stats": self.stats.copy(),
            "orchestrator_stats": self.orchestrator.get_statistics(),
            "config": self.config.copy(),
            "cache_size": len(self._cache),
            "status": self.state.status.value,
            "is_healthy": self.is_healthy,
            "performance_stats": {
                "chain_execution": performance_monitor.get_operation_statistics("intelligent_chain_execution"),
                "overall_system": performance_monitor.get_overall_statistics(),
                "optimization_recommendations": performance_monitor.get_optimization_recommendations()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            # Check orchestrator health
            orchestrator_health = await self.orchestrator.health_check()
            
            # Check agent health
            agent_healthy = (
                self.state.status in [AgentStatus.IDLE, AgentStatus.RUNNING] and 
                self.is_healthy and 
                orchestrator_health.get("healthy", False)
            )
            
            return {
                "healthy": agent_healthy,
                "status": self.state.status.value,
                "agent_id": self.agent_id,
                "orchestrator": orchestrator_health,
                "statistics": self.get_agent_statistics(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ======================
    # PRIVATE HELPER METHODS
    # ======================
    
    def _validate_query(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate query before processing."""
        if not query or len(query.strip()) == 0:
            return {"valid": False, "reason": "Empty query"}
        
        if len(query) > 2000:
            return {"valid": False, "reason": "Query too long (max 2000 characters)"}
        
        # Add more validation as needed
        return {"valid": True}
    
    def _post_process_chain_result(
        self,
        chain_result: ChainResult,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post-process chain results for API response."""
        # Filter out sensitive information if needed
        processed_results = []
        
        for result in chain_result.results:
            processed_result = {
                "step_id": result.step_id,
                "operation_type": result.operation_type.value,
                "success": result.success,
                "duration": result.duration,
                "metadata": result.metadata
            }
            
            # Include data if successful
            if result.success and result.data:
                processed_result["data"] = result.data
            
            # Include error if failed
            if not result.success and result.error:
                processed_result["error"] = result.error
            
            processed_results.append(processed_result)
        
        return {
            "success": chain_result.success,
            "query": chain_result.query,
            "intent": chain_result.intent.value,
            "plan_id": chain_result.plan_id,
            "total_duration": chain_result.total_duration,
            "operations": processed_results,
            "final_data": chain_result.final_data,
            "metadata": chain_result.metadata,
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error": chain_result.error if not chain_result.success else None
        }
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key for query and context."""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        return f"query:{hash(query)}:context:{hash(context_str)}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache if still valid."""
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            
            # Check if cache is still valid
            age = (datetime.utcnow() - timestamp).total_seconds()
            if age < self._cache_ttl:
                return cached_data
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        
        return None
    
    def _store_in_cache(self, cache_key: str, result: Dict[str, Any]):
        """Store result in cache."""
        self._cache[cache_key] = (result, datetime.utcnow())
        
        # Simple cache cleanup - remove oldest entries if cache gets too large
        if len(self._cache) > 100:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
    
    def _update_agent_stats(self, chain_result: ChainResult, start_time: datetime):
        """Update agent statistics."""
        self.stats["total_queries"] += 1
        
        if chain_result.success:
            self.stats["successful_chains"] += 1
        else:
            self.stats["failed_chains"] += 1
        
        # Update average duration
        total_queries = self.stats["total_queries"]
        current_avg = self.stats["avg_chain_duration"]
        new_duration = chain_result.total_duration
        
        self.stats["avg_chain_duration"] = (
            (current_avg * (total_queries - 1) + new_duration) / total_queries
        )
        
        # Update intent distribution
        intent_str = chain_result.intent.value
        self.stats["intent_distribution"][intent_str] = (
            self.stats["intent_distribution"].get(intent_str, 0) + 1
        )
        
        # Update average confidence (from plan metadata)
        plan_confidence = chain_result.metadata.get("plan_confidence", 0.5)
        current_confidence = self.stats["avg_confidence"]
        
        self.stats["avg_confidence"] = (
            (current_confidence * (total_queries - 1) + plan_confidence) / total_queries
        ) 
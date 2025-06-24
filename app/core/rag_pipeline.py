"""
RAG Pipeline Orchestrator for coordinating all agents in the RAG pipeline.

This orchestrator handles:
- Agent coordination and communication
- Pipeline flow control and error handling
- Performance monitoring and optimization
- Async processing with WebSocket support
- Fallback strategies for agent failures
- Real-time pipeline status updates
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from datetime import datetime
from enum import Enum
import uuid

from ..agents.query_rewriter import QueryRewritingAgent
from ..agents.context_decision import ContextDecisionAgent
from ..agents.source_retrieval import SourceRetrievalAgent
from ..agents.answer_generation import AnswerGenerationAgent
from ..agents.registry import AgentRegistry
from ..agents.metrics import AgentMetrics

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Enumeration for pipeline stages."""
    QUERY_REWRITING = "query_rewriting"
    CONTEXT_DECISION = "context_decision"
    SOURCE_RETRIEVAL = "source_retrieval"
    ANSWER_GENERATION = "answer_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStatus(Enum):
    """Enumeration for pipeline status."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineResult:
    """Class representing a pipeline execution result."""
    
    def __init__(
        self,
        request_id: str,
        query: str,
        status: PipelineStatus,
        final_response: Optional[Dict[str, Any]] = None,
        stage_results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.request_id = request_id
        self.query = query
        self.status = status
        self.final_response = final_response
        self.stage_results = stage_results or {}
        self.error = error
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.completed_at = None
        self.total_duration = 0.0
    
    def complete(self, final_response: Dict[str, Any]) -> None:
        """Mark the pipeline as completed."""
        self.status = PipelineStatus.COMPLETED
        self.final_response = final_response
        self.completed_at = datetime.utcnow()
        self.total_duration = (self.completed_at - self.created_at).total_seconds()
    
    def fail(self, error: str) -> None:
        """Mark the pipeline as failed."""
        self.status = PipelineStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()
        self.total_duration = (self.completed_at - self.created_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "query": self.query,
            "status": self.status.value,
            "final_response": self.final_response,
            "stage_results": self.stage_results,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration": self.total_duration
        }


class RAGPipelineOrchestrator:
    """
    Orchestrator for the RAG pipeline that coordinates all agents.
    
    Pipeline Flow:
    1. Query Rewriting Agent - Optimizes and normalizes the query
    2. Context Decision Agent - Determines if context retrieval is needed
    3. Source Retrieval Agent - Retrieves relevant sources (if needed)
    4. Answer Generation Agent - Generates final response with citations
    """
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        agent_metrics: AgentMetrics,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the RAG Pipeline Orchestrator."""
        self.agent_registry = agent_registry
        self.agent_metrics = agent_metrics
        self.config = config or {}
        
        # Pipeline configuration
        self.max_pipeline_duration = self.config.get("max_pipeline_duration", 30.0)  # 30 seconds
        self.enable_fallbacks = self.config.get("enable_fallbacks", True)
        self.enable_caching = self.config.get("enable_caching", True)
        self.enable_streaming = self.config.get("enable_streaming", True)
        self.parallel_processing = self.config.get("parallel_processing", False)
        
        # Agent configurations
        self.agent_configs = {
            "query_rewriter": self.config.get("query_rewriter", {}),
            "context_decision": self.config.get("context_decision", {}),
            "source_retrieval": self.config.get("source_retrieval", {}),
            "answer_generation": self.config.get("answer_generation", {})
        }
        
        # Pipeline state
        self.active_pipelines: Dict[str, PipelineResult] = {}
        self.pipeline_cache: Dict[str, PipelineResult] = {}
        self.cache_ttl = 600  # 10 minutes
        
        # Performance tracking
        self.pipeline_stats = {
            "total_pipelines": 0,
            "successful_pipelines": 0,
            "failed_pipelines": 0,
            "avg_duration": 0.0,
            "stage_performance": {
                "query_rewriting": {"count": 0, "avg_duration": 0.0, "success_rate": 0.0},
                "context_decision": {"count": 0, "avg_duration": 0.0, "success_rate": 0.0},
                "source_retrieval": {"count": 0, "avg_duration": 0.0, "success_rate": 0.0},
                "answer_generation": {"count": 0, "avg_duration": 0.0, "success_rate": 0.0}
            }
        }
        
        logger.info("RAG Pipeline Orchestrator initialized")
    
    async def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: User query to process
            conversation_history: Previous conversation messages
            user_context: Additional user context
            pipeline_config: Pipeline-specific configuration overrides
            
        Returns:
            PipelineResult with final response and metadata
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create pipeline result
        result = PipelineResult(
            request_id=request_id,
            query=query,
            status=PipelineStatus.PROCESSING,
            metadata={
                "conversation_history": conversation_history or [],
                "user_context": user_context or {},
                "pipeline_config": pipeline_config or {}
            }
        )
        
        self.active_pipelines[request_id] = result
        
        try:
            logger.info(f"Starting RAG pipeline for request {request_id}: '{query[:50]}...'")
            
            # Check cache first
            if self.enable_caching:
                cached_result = self._get_cached_result(query, conversation_history)
                if cached_result:
                    logger.info(f"Cache hit for request {request_id}")
                    result.complete(cached_result.final_response)
                    return result
            
            # Stage 1: Query Rewriting
            rewritten_query = await self._execute_query_rewriting(
                query, conversation_history, result
            )
            
            # Stage 2: Context Decision
            context_needed = await self._execute_context_decision(
                rewritten_query, conversation_history, result
            )
            
            # Stage 3: Source Retrieval (if needed)
            sources = []
            if context_needed:
                sources = await self._execute_source_retrieval(
                    rewritten_query, conversation_history, result
                )
            else:
                logger.info(f"Context not needed for request {request_id}, skipping source retrieval")
                result.stage_results["source_retrieval"] = {
                    "skipped": True,
                    "reason": "Context not needed",
                    "sources": []
                }
            # Stage 4: Answer Generation
            final_response = await self._execute_answer_generation(
                rewritten_query, sources, conversation_history, result
            )
            
            # Complete pipeline
            result.complete(final_response)
            
            # Cache result
            if self.enable_caching:
                self._cache_result(query, conversation_history, result)
            
            # Update statistics
            self._update_pipeline_stats(result, time.time() - start_time)
            
            logger.info(f"RAG pipeline completed for request {request_id} in {result.total_duration:.3f}s")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(f"RAG pipeline failed for request {request_id}: {error_msg}")
            result.fail(error_msg)
            self._update_pipeline_stats(result, time.time() - start_time)
        
        finally:
            # Clean up active pipeline
            if request_id in self.active_pipelines:
                del self.active_pipelines[request_id]
        
        return result
    
    async def stream_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a query through the RAG pipeline with streaming updates.
        
        Yields:
            Dictionary updates for each pipeline stage
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"Starting streaming RAG pipeline for request {request_id}")
            
            # Initial status
            yield {
                "request_id": request_id,
                "stage": PipelineStage.QUERY_REWRITING.value,
                "status": "starting",
                "message": "Starting query rewriting...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Stage 1: Query Rewriting
            yield {
                "request_id": request_id,
                "stage": PipelineStage.QUERY_REWRITING.value,
                "status": "processing",
                "message": "Rewriting and optimizing query...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            rewritten_query = await self._execute_query_rewriting_stream(query, conversation_history)
            
            yield {
                "request_id": request_id,
                "stage": PipelineStage.QUERY_REWRITING.value,
                "status": "completed",
                "result": {"rewritten_query": rewritten_query},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Stage 2: Context Decision
            yield {
                "request_id": request_id,
                "stage": PipelineStage.CONTEXT_DECISION.value,
                "status": "processing",
                "message": "Determining context necessity...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            context_needed = await self._execute_context_decision_stream(rewritten_query, conversation_history)
            
            yield {
                "request_id": request_id,
                "stage": PipelineStage.CONTEXT_DECISION.value,
                "status": "completed",
                "result": {"context_needed": context_needed},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Stage 3: Source Retrieval (if needed)
            sources = []
            if context_needed:
                yield {
                    "request_id": request_id,
                    "stage": PipelineStage.SOURCE_RETRIEVAL.value,
                    "status": "processing",
                    "message": "Retrieving relevant sources...",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                sources = await self._execute_source_retrieval_stream(rewritten_query, conversation_history)
                
                yield {
                    "request_id": request_id,
                    "stage": PipelineStage.SOURCE_RETRIEVAL.value,
                    "status": "completed",
                    "result": {"sources_count": len(sources)},
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                yield {
                    "request_id": request_id,
                    "stage": PipelineStage.SOURCE_RETRIEVAL.value,
                    "status": "skipped",
                    "message": "Context not needed, skipping source retrieval",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Stage 4: Answer Generation
            yield {
                "request_id": request_id,
                "stage": PipelineStage.ANSWER_GENERATION.value,
                "status": "processing",
                "message": "Generating response...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Stream answer generation
            final_response = None
            async for chunk in self._execute_answer_generation_stream(rewritten_query, sources, conversation_history):
                if isinstance(chunk, dict) and "final_response" in chunk:
                    final_response = chunk["final_response"]
                    break
                else:
                    yield {
                        "request_id": request_id,
                        "stage": PipelineStage.ANSWER_GENERATION.value,
                        "status": "streaming",
                        "chunk": chunk,
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            # Final completion
            yield {
                "request_id": request_id,
                "stage": PipelineStage.COMPLETED.value,
                "status": "completed",
                "final_response": final_response,
                "total_duration": time.time() - start_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Streaming pipeline failed for request {request_id}: {str(e)}")
            yield {
                "request_id": request_id,
                "stage": PipelineStage.FAILED.value,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _execute_query_rewriting(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        result: PipelineResult
    ) -> str:
        """Execute the query rewriting stage."""
        stage_start = time.time()
        
        try:
            # Get or create query rewriter agent
            agent = await self._get_or_create_agent("query_rewriter", QueryRewritingAgent)
            
            # Process query
            agent_result = await agent.process({
                "query": query,
                "conversation_history": conversation_history or []
            })
            
            if not agent_result.success:
                raise Exception(f"Query rewriting failed: {agent_result.error}")
            
            rewritten_query = agent_result.data.get("rewritten_query", query)
            
            # Store stage result
            result.stage_results["query_rewriting"] = {
                "original_query": query,
                "rewritten_query": rewritten_query,
                "duration": time.time() - stage_start,
                "agent_id": agent.agent_id
            }
            
            return rewritten_query
            
        except Exception as e:
            self._handle_stage_error("query_rewriting", str(e), result)
            # Fallback to original query
            return query
    
    async def _execute_context_decision(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        result: PipelineResult
    ) -> bool:
        """Execute the context decision stage."""
        stage_start = time.time()
        
        try:
            # Get or create context decision agent
            agent = await self._get_or_create_agent("context_decision", ContextDecisionAgent)
            
            # Process query
            agent_result = await agent.process({
                "query": query,
                "conversation_history": conversation_history or []
            })
            
            if not agent_result.success:
                raise Exception(f"Context decision failed: {agent_result.error}")
            
            context_needed = agent_result.data.get("context_needed", True)
            
            # Store stage result
            result.stage_results["context_decision"] = {
                "query": query,
                "context_needed": context_needed,
                "confidence": agent_result.data.get("confidence", 0.0),
                "reasoning": agent_result.data.get("reasoning", ""),
                "duration": time.time() - stage_start,
                "agent_id": agent.agent_id
            }
            
            return context_needed
            
        except Exception as e:
            self._handle_stage_error("context_decision", str(e), result)
            # Fallback to always needing context
            return True
    
    async def _execute_source_retrieval(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        result: PipelineResult
    ) -> List[Dict[str, Any]]:
        """Execute the source retrieval stage."""
        stage_start = time.time()
        
        try:
            # Get or create source retrieval agent
            agent = await self._get_or_create_agent("source_retrieval", SourceRetrievalAgent)
            
            # Process query
            agent_result = await agent.process({
                "query": query,
                "conversation_history": conversation_history or [],
                "max_sources": 10
            })
            
            if not agent_result.success:
                raise Exception(f"Source retrieval failed: {agent_result.error}")
            
            sources = agent_result.data.get("sources", [])
            
            # Store stage result
            result.stage_results["source_retrieval"] = {
                "query": query,
                "sources_count": len(sources),
                "sources": sources,
                "retrieval_strategy": agent_result.data.get("strategy_used", "unknown"),
                "duration": time.time() - stage_start,
                "agent_id": agent.agent_id
            }
            
            return sources
            
        except Exception as e:
            self._handle_stage_error("source_retrieval", str(e), result)
            # Fallback to empty sources
            return []
    
    async def _execute_answer_generation(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]],
        result: PipelineResult
    ) -> Dict[str, Any]:
        """Execute the answer generation stage."""
        stage_start = time.time()
        
        try:
            # Get or create answer generation agent
            agent = await self._get_or_create_agent("answer_generation", AnswerGenerationAgent)
            
            # Process query
            agent_result = await agent.process({
                "query": query,
                "sources": sources,
                "conversation_history": conversation_history or [],
                "generation_config": {}
            })
            
            if not agent_result.success:
                raise Exception(f"Answer generation failed: {agent_result.error}")
            
            final_response = agent_result.data
            
            # Store stage result
            result.stage_results["answer_generation"] = {
                "query": query,
                "sources_used": len(sources),
                "response_length": len(final_response.get("response", {}).get("content", "")),
                "citations_count": len(final_response.get("response", {}).get("citations", [])),
                "duration": time.time() - stage_start,
                "agent_id": agent.agent_id
            }
            
            return final_response
            
        except Exception as e:
            self._handle_stage_error("answer_generation", str(e), result)
            # Fallback response
            return {
                "query": query,
                "response": {
                    "content": f"I apologize, but I encountered an error while generating a response to your query: '{query}'. Please try again.",
                    "citations": [],
                    "quality": {"overall_quality": 0.0},
                    "format_type": "markdown"
                },
                "sources_used": 0,
                "generation_metadata": {
                    "fallback": True,
                    "error": str(e)
                }
            }
    
    async def _execute_query_rewriting_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Execute query rewriting for streaming pipeline."""
        try:
            agent = await self._get_or_create_agent("query_rewriter", QueryRewritingAgent)
            agent_result = await agent.process({
                "query": query,
                "conversation_history": conversation_history or []
            })
            return agent_result.data.get("rewritten_query", query) if agent_result.success else query
        except Exception:
            return query
    
    async def _execute_context_decision_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> bool:
        """Execute context decision for streaming pipeline."""
        try:
            agent = await self._get_or_create_agent("context_decision", ContextDecisionAgent)
            agent_result = await agent.process({
                "query": query,
                "conversation_history": conversation_history or []
            })
            return agent_result.data.get("context_needed", True) if agent_result.success else True
        except Exception:
            return True
    
    async def _execute_source_retrieval_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Execute source retrieval for streaming pipeline."""
        try:
            agent = await self._get_or_create_agent("source_retrieval", SourceRetrievalAgent)
            agent_result = await agent.process({
                "query": query,
                "conversation_history": conversation_history or [],
                "max_sources": 10
            })
            return agent_result.data.get("sources", []) if agent_result.success else []
        except Exception:
            return []
    
    async def _execute_answer_generation_stream(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> AsyncGenerator[Any, None]:
        """Execute answer generation for streaming pipeline."""
        try:
            agent = await self._get_or_create_agent("answer_generation", AnswerGenerationAgent)
            
            # Use streaming if available
            if hasattr(agent, 'stream_response'):
                async for chunk in agent.stream_response(
                    query=query,
                    sources=sources,
                    conversation_history=conversation_history or []
                ):
                    yield chunk
            else:
                # Fallback to regular processing
                agent_result = await agent.process({
                    "query": query,
                    "sources": sources,
                    "conversation_history": conversation_history or []
                })
                yield {"final_response": agent_result.data if agent_result.success else None}
                
        except Exception as e:
            yield {"final_response": None, "error": str(e)}
    
    async def _get_or_create_agent(self, agent_type: str, agent_class) -> Any:
        """Get existing agent or create new one."""
        agents = self.agent_registry.get_agents_by_type(agent_type)
        
        if agents:
            # Use existing agent
            agent = agents[0]
            if not agent.is_running:
                await agent.start()
            return agent
        else:
            # Create new agent
            config = self.agent_configs.get(agent_type, {})
            agent = await self.agent_registry.create_agent(
                agent_type=agent_type,
                config=config,
                auto_start=True
            )
            return agent
    
    def _handle_stage_error(self, stage: str, error: str, result: PipelineResult) -> None:
        """Handle stage-specific errors."""
        logger.error(f"Stage {stage} failed: {error}")
        
        result.stage_results[stage] = {
            "error": error,
            "fallback_used": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if not self.enable_fallbacks:
            raise Exception(f"Stage {stage} failed and fallbacks disabled: {error}")
    
    def _get_cached_result(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> Optional[PipelineResult]:
        """Get cached pipeline result if available."""
        cache_key = self._generate_cache_key(query, conversation_history)
        
        if cache_key in self.pipeline_cache:
            cached_result = self.pipeline_cache[cache_key]
            # Check if cache is still valid
            if (datetime.utcnow() - cached_result.created_at).total_seconds() < self.cache_ttl:
                return cached_result
            else:
                del self.pipeline_cache[cache_key]
        
        return None
    
    def _cache_result(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        result: PipelineResult
    ) -> None:
        """Cache pipeline result."""
        cache_key = self._generate_cache_key(query, conversation_history)
        self.pipeline_cache[cache_key] = result
        
        # Clean old cache entries
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, cached_result in self.pipeline_cache.items()
            if (current_time - cached_result.created_at).total_seconds() > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.pipeline_cache[key]
    
    def _generate_cache_key(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Generate cache key for pipeline result."""
        import hashlib
        
        # Create hash based on query and recent conversation
        recent_history = (conversation_history or [])[-3:]  # Last 3 messages
        key_data = f"{query}:{json.dumps(recent_history, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _update_pipeline_stats(self, result: PipelineResult, duration: float) -> None:
        """Update pipeline performance statistics."""
        self.pipeline_stats["total_pipelines"] += 1
        
        if result.status == PipelineStatus.COMPLETED:
            self.pipeline_stats["successful_pipelines"] += 1
        else:
            self.pipeline_stats["failed_pipelines"] += 1
        
        # Update average duration
        total = self.pipeline_stats["total_pipelines"]
        current_avg = self.pipeline_stats["avg_duration"]
        self.pipeline_stats["avg_duration"] = (current_avg * (total - 1) + duration) / total
        
        # Update stage performance
        for stage, stage_data in result.stage_results.items():
            if stage in self.pipeline_stats["stage_performance"]:
                stage_stats = self.pipeline_stats["stage_performance"][stage]
                stage_stats["count"] += 1
                
                if "duration" in stage_data:
                    stage_duration = stage_data["duration"]
                    current_stage_avg = stage_stats["avg_duration"]
                    stage_stats["avg_duration"] = (
                        (current_stage_avg * (stage_stats["count"] - 1) + stage_duration) / stage_stats["count"]
                    )
                
                # Update success rate
                if "error" not in stage_data:
                    successes = stage_stats["count"] * stage_stats["success_rate"] + 1
                    stage_stats["success_rate"] = successes / stage_stats["count"]
                else:
                    successes = stage_stats["count"] * stage_stats["success_rate"]
                    stage_stats["success_rate"] = successes / stage_stats["count"]
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        return {
            "active_pipelines": len(self.active_pipelines),
            "cached_results": len(self.pipeline_cache),
            "statistics": self.pipeline_stats,
            "configuration": {
                "max_pipeline_duration": self.max_pipeline_duration,
                "enable_fallbacks": self.enable_fallbacks,
                "enable_caching": self.enable_caching,
                "enable_streaming": self.enable_streaming
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_active_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently active pipelines."""
        return {
            request_id: {
                "query": result.query[:100] + "..." if len(result.query) > 100 else result.query,
                "status": result.status.value,
                "created_at": result.created_at.isoformat(),
                "duration": (datetime.utcnow() - result.created_at).total_seconds(),
                "current_stage": self._get_current_stage(result)
            }
            for request_id, result in self.active_pipelines.items()
        }
    
    def _get_current_stage(self, result: PipelineResult) -> str:
        """Determine the current stage of a pipeline."""
        stages = ["query_rewriting", "context_decision", "source_retrieval", "answer_generation"]
        
        for stage in stages:
            if stage not in result.stage_results:
                return stage
        
        return "completed" if result.status == PipelineStatus.COMPLETED else "unknown" 
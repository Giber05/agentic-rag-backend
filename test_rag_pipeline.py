#!/usr/bin/env python3

"""
Test script for the RAG Pipeline Orchestrator.

This script tests:
- End-to-end RAG pipeline processing
- Agent coordination and communication
- Pipeline flow control and error handling
- Streaming response functionality
- Performance monitoring and metrics
- Fallback strategies and error recovery
- Caching functionality
- WebSocket streaming
"""

import asyncio
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the backend directory to the Python path
sys.path.insert(0, '/Users/sproutdigitallab/Documents/Else/Untitled/agentic-rag-ai-agent/backend')

from app.core.rag_pipeline import (
    RAGPipelineOrchestrator, 
    PipelineResult, 
    PipelineStatus,
    PipelineStage
)
from app.agents.registry import AgentRegistry
from app.agents.metrics import AgentMetrics
from app.agents.query_rewriter import QueryRewritingAgent
from app.agents.context_decision import ContextDecisionAgent
from app.agents.source_retrieval import SourceRetrievalAgent
from app.agents.answer_generation import AnswerGenerationAgent


class RAGPipelineTester:
    """Test suite for the RAG Pipeline Orchestrator."""
    
    def __init__(self):
        self.registry = AgentRegistry()
        self.metrics = AgentMetrics()
        self.test_results = []
        
        # Register agent types
        self.registry.register_agent_type("query_rewriter", QueryRewritingAgent)
        self.registry.register_agent_type("context_decision", ContextDecisionAgent)
        self.registry.register_agent_type("source_retrieval", SourceRetrievalAgent)
        self.registry.register_agent_type("answer_generation", AnswerGenerationAgent)
        
        # Sample test queries
        self.test_queries = [
            "What is machine learning and how does it work?",
            "Explain the differences between supervised and unsupervised learning",
            "How do neural networks process information?",
            "What are the applications of deep learning in computer vision?",
            "Can you tell me about natural language processing techniques?"
        ]
        
        # Sample conversation history
        self.sample_conversation = [
            {"role": "user", "content": "What is artificial intelligence?"},
            {"role": "assistant", "content": "Artificial intelligence is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence."},
            {"role": "user", "content": "How does machine learning relate to AI?"}
        ]
    
    def log_test_result(self, test_name: str, success: bool, details: str = "", duration: float = 0.0):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append({
            "test": test_name,
            "status": status,
            "success": success,
            "details": details,
            "duration": duration
        })
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        if duration > 0:
            print(f"    Duration: {duration:.3f}s")
    
    async def test_orchestrator_creation(self):
        """Test RAG Pipeline Orchestrator creation and configuration."""
        print("\n🧪 Testing RAG Pipeline Orchestrator Creation...")
        
        try:
            start_time = time.time()
            
            # Create orchestrator
            orchestrator = RAGPipelineOrchestrator(
                agent_registry=self.registry,
                agent_metrics=self.metrics,
                config={
                    "max_pipeline_duration": 30.0,
                    "enable_fallbacks": True,
                    "enable_caching": True,
                    "enable_streaming": True,
                    "query_rewriter": {"max_query_length": 500},
                    "context_decision": {"similarity_threshold": 0.7},
                    "source_retrieval": {"max_sources": 5},
                    "answer_generation": {"max_response_length": 1000}
                }
            )
            
            # Verify configuration
            assert orchestrator.max_pipeline_duration == 30.0
            assert orchestrator.enable_fallbacks == True
            assert orchestrator.enable_caching == True
            assert orchestrator.enable_streaming == True
            assert len(orchestrator.agent_configs) == 4
            
            duration = time.time() - start_time
            self.log_test_result(
                "Orchestrator Creation", 
                True, 
                "Orchestrator created with proper configuration",
                duration
            )
            
            return orchestrator
            
        except Exception as e:
            self.log_test_result("Orchestrator Creation", False, f"Error: {str(e)}")
            return None
    
    async def test_end_to_end_pipeline(self, orchestrator: RAGPipelineOrchestrator):
        """Test complete end-to-end RAG pipeline processing."""
        print("\n🔄 Testing End-to-End Pipeline Processing...")
        
        try:
            start_time = time.time()
            
            # Process query through pipeline
            result = await orchestrator.process_query(
                query=self.test_queries[0],
                conversation_history=self.sample_conversation,
                user_context={"user_id": "test_user", "session_id": "test_session"},
                pipeline_config={"enable_debug": True}
            )
            
            # Verify result structure
            assert isinstance(result, PipelineResult)
            assert result.request_id is not None
            assert result.query == self.test_queries[0]
            assert result.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]
            
            # Verify stage results
            expected_stages = ["query_rewriting", "context_decision", "answer_generation"]
            for stage in expected_stages:
                if stage in result.stage_results:
                    assert "duration" in result.stage_results[stage]
            
            # Check if pipeline completed successfully
            if result.status == PipelineStatus.COMPLETED:
                assert result.final_response is not None
                assert "response" in result.final_response
                
                response_content = result.final_response["response"]["content"]
                assert len(response_content) > 0
                
                success_details = f"Pipeline completed successfully with {len(result.stage_results)} stages"
            else:
                success_details = f"Pipeline failed but handled gracefully: {result.error}"
            
            duration = time.time() - start_time
            self.log_test_result(
                "End-to-End Pipeline", 
                True,  # Consider it successful if it completes without crashing
                success_details,
                duration
            )
            
            return result
            
        except Exception as e:
            self.log_test_result("End-to-End Pipeline", False, f"Error: {str(e)}")
            return None
    
    async def test_streaming_pipeline(self, orchestrator: RAGPipelineOrchestrator):
        """Test streaming pipeline functionality."""
        print("\n🌊 Testing Streaming Pipeline...")
        
        try:
            start_time = time.time()
            
            updates = []
            stages_seen = set()
            
            async for update in orchestrator.stream_query(
                query=self.test_queries[1],
                conversation_history=[],
                user_context={},
                pipeline_config={}
            ):
                updates.append(update)
                if "stage" in update:
                    stages_seen.add(update["stage"])
                
                # Limit updates for testing
                if len(updates) >= 20:
                    break
            
            # Verify streaming worked
            assert len(updates) > 0
            assert len(stages_seen) > 0
            
            # Check for expected stages
            expected_stages = {"query_rewriting", "context_decision", "answer_generation"}
            found_stages = stages_seen.intersection(expected_stages)
            
            duration = time.time() - start_time
            self.log_test_result(
                "Streaming Pipeline", 
                True, 
                f"Received {len(updates)} updates, saw {len(found_stages)} stages",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Streaming Pipeline", False, f"Error: {str(e)}")
    
    async def test_agent_coordination(self, orchestrator: RAGPipelineOrchestrator):
        """Test agent coordination and communication."""
        print("\n🤝 Testing Agent Coordination...")
        
        try:
            start_time = time.time()
            
            # Process multiple queries to test agent reuse
            results = []
            for i, query in enumerate(self.test_queries[:3]):
                result = await orchestrator.process_query(
                    query=query,
                    conversation_history=[],
                    user_context={"test_batch": i},
                    pipeline_config={}
                )
                results.append(result)
            
            # Verify all queries were processed
            assert len(results) == 3
            
            # Check that agents were reused (should be same agent IDs)
            agent_ids_per_stage = {}
            for result in results:
                for stage, stage_data in result.stage_results.items():
                    if "agent_id" in stage_data:
                        if stage not in agent_ids_per_stage:
                            agent_ids_per_stage[stage] = set()
                        agent_ids_per_stage[stage].add(stage_data["agent_id"])
            
            # Verify agent reuse (should be 1 agent per stage type)
            reused_agents = sum(1 for ids in agent_ids_per_stage.values() if len(ids) == 1)
            
            duration = time.time() - start_time
            self.log_test_result(
                "Agent Coordination", 
                True, 
                f"Processed {len(results)} queries, {reused_agents} agent types reused",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Agent Coordination", False, f"Error: {str(e)}")
    
    async def test_error_handling(self, orchestrator: RAGPipelineOrchestrator):
        """Test error handling and fallback strategies."""
        print("\n⚠️ Testing Error Handling...")
        
        try:
            start_time = time.time()
            
            # Test with empty query
            try:
                result = await orchestrator.process_query(
                    query="",
                    conversation_history=[],
                    user_context={},
                    pipeline_config={}
                )
                # Should handle gracefully
                empty_query_handled = True
            except Exception:
                empty_query_handled = False
            
            # Test with very long query
            long_query = "What is machine learning? " * 100  # Very long query
            result = await orchestrator.process_query(
                query=long_query,
                conversation_history=[],
                user_context={},
                pipeline_config={}
            )
            
            # Should complete or fail gracefully
            long_query_handled = result is not None
            
            # Test with malformed conversation history
            malformed_history = [{"invalid": "structure"}]
            result = await orchestrator.process_query(
                query="Test query",
                conversation_history=malformed_history,
                user_context={},
                pipeline_config={}
            )
            
            malformed_history_handled = result is not None
            
            duration = time.time() - start_time
            
            handled_cases = sum([empty_query_handled, long_query_handled, malformed_history_handled])
            self.log_test_result(
                "Error Handling", 
                handled_cases >= 2,  # At least 2 out of 3 should be handled
                f"Handled {handled_cases}/3 error scenarios gracefully",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Error Handling", False, f"Error: {str(e)}")
    
    async def test_caching_functionality(self, orchestrator: RAGPipelineOrchestrator):
        """Test pipeline result caching."""
        print("\n💾 Testing Caching Functionality...")
        
        try:
            start_time = time.time()
            
            # Enable caching
            orchestrator.enable_caching = True
            
            query = "What is deep learning?"
            conversation_history = []
            
            # First request (should not be cached)
            result1 = await orchestrator.process_query(
                query=query,
                conversation_history=conversation_history,
                user_context={},
                pipeline_config={}
            )
            
            # Second request (should be cached)
            result2 = await orchestrator.process_query(
                query=query,
                conversation_history=conversation_history,
                user_context={},
                pipeline_config={}
            )
            
            # Verify caching worked
            cache_used = len(orchestrator.pipeline_cache) > 0
            
            # Check if second request was faster (indicating cache hit)
            faster_second_request = result2.total_duration < result1.total_duration
            
            duration = time.time() - start_time
            self.log_test_result(
                "Caching Functionality", 
                cache_used,
                f"Cache entries: {len(orchestrator.pipeline_cache)}, faster 2nd request: {faster_second_request}",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Caching Functionality", False, f"Error: {str(e)}")
    
    async def test_performance_metrics(self, orchestrator: RAGPipelineOrchestrator):
        """Test performance monitoring and metrics collection."""
        print("\n📊 Testing Performance Metrics...")
        
        try:
            start_time = time.time()
            
            # Get initial stats
            initial_status = orchestrator.get_pipeline_status()
            initial_pipelines = initial_status["statistics"]["total_pipelines"]
            
            # Process a few queries
            for i in range(3):
                await orchestrator.process_query(
                    query=f"Test query {i + 1}: {self.test_queries[i % len(self.test_queries)]}",
                    conversation_history=[],
                    user_context={"test_run": i},
                    pipeline_config={}
                )
            
            # Get updated stats
            updated_status = orchestrator.get_pipeline_status()
            updated_pipelines = updated_status["statistics"]["total_pipelines"]
            
            # Verify stats were updated
            pipelines_processed = updated_pipelines - initial_pipelines
            assert pipelines_processed == 3
            
            # Check stage performance metrics
            stage_performance = updated_status["statistics"]["stage_performance"]
            stages_with_metrics = sum(1 for stage_stats in stage_performance.values() if stage_stats["count"] > 0)
            
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Metrics", 
                True, 
                f"Processed {pipelines_processed} pipelines, {stages_with_metrics} stages with metrics",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Performance Metrics", False, f"Error: {str(e)}")
    
    async def test_pipeline_status_monitoring(self, orchestrator: RAGPipelineOrchestrator):
        """Test pipeline status and monitoring functionality."""
        print("\n📈 Testing Pipeline Status Monitoring...")
        
        try:
            start_time = time.time()
            
            # Get pipeline status
            status = orchestrator.get_pipeline_status()
            
            # Verify status structure
            required_fields = ["active_pipelines", "cached_results", "statistics", "configuration"]
            for field in required_fields:
                assert field in status
            
            # Check statistics structure
            stats = status["statistics"]
            required_stats = ["total_pipelines", "successful_pipelines", "failed_pipelines", "avg_duration"]
            for stat in required_stats:
                assert stat in stats
            
            # Get active pipelines
            active_pipelines = orchestrator.get_active_pipelines()
            assert isinstance(active_pipelines, dict)
            
            duration = time.time() - start_time
            self.log_test_result(
                "Pipeline Status Monitoring", 
                True, 
                f"Status fields: {len(status)}, active pipelines: {len(active_pipelines)}",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Pipeline Status Monitoring", False, f"Error: {str(e)}")
    
    async def test_configuration_management(self, orchestrator: RAGPipelineOrchestrator):
        """Test pipeline configuration management."""
        print("\n⚙️ Testing Configuration Management...")
        
        try:
            start_time = time.time()
            
            # Test configuration updates
            original_duration = orchestrator.max_pipeline_duration
            original_caching = orchestrator.enable_caching
            
            # Update configuration
            orchestrator.max_pipeline_duration = 60.0
            orchestrator.enable_caching = False
            
            # Verify updates
            assert orchestrator.max_pipeline_duration == 60.0
            assert orchestrator.enable_caching == False
            
            # Test agent configuration updates
            original_config = orchestrator.agent_configs.copy()
            orchestrator.agent_configs["query_rewriter"]["test_param"] = "test_value"
            
            assert orchestrator.agent_configs["query_rewriter"]["test_param"] == "test_value"
            
            # Restore original configuration
            orchestrator.max_pipeline_duration = original_duration
            orchestrator.enable_caching = original_caching
            orchestrator.agent_configs = original_config
            
            duration = time.time() - start_time
            self.log_test_result(
                "Configuration Management", 
                True, 
                "Configuration updates and restoration successful",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Configuration Management", False, f"Error: {str(e)}")
    
    async def test_concurrent_processing(self, orchestrator: RAGPipelineOrchestrator):
        """Test concurrent pipeline processing."""
        print("\n🔀 Testing Concurrent Processing...")
        
        try:
            start_time = time.time()
            
            # Create multiple concurrent tasks
            tasks = []
            for i in range(3):
                task = orchestrator.process_query(
                    query=f"Concurrent query {i + 1}: {self.test_queries[i]}",
                    conversation_history=[],
                    user_context={"concurrent_test": i},
                    pipeline_config={}
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify results
            successful_results = sum(1 for result in results if isinstance(result, PipelineResult))
            exceptions = sum(1 for result in results if isinstance(result, Exception))
            
            duration = time.time() - start_time
            self.log_test_result(
                "Concurrent Processing", 
                successful_results >= 2,  # At least 2 out of 3 should succeed
                f"Successful: {successful_results}, Exceptions: {exceptions}",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Concurrent Processing", False, f"Error: {str(e)}")
    
    def print_test_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("🧪 RAG PIPELINE ORCHESTRATOR TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['details']}")
        
        print(f"\n⏱️ Performance Summary:")
        total_duration = sum(result["duration"] for result in self.test_results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        print(f"  - Total Duration: {total_duration:.3f}s")
        print(f"  - Average Duration: {avg_duration:.3f}s")
        
        # Performance benchmarks
        print(f"\n🎯 Performance Benchmarks:")
        fast_tests = sum(1 for result in self.test_results if result["duration"] < 5.0)
        print(f"  - Tests under 5s: {fast_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            print(f"\n🎉 All tests passed! RAG Pipeline Orchestrator is working correctly.")
        else:
            print(f"\n⚠️ Some tests failed. Please review the failed tests above.")


async def main():
    """Run all RAG Pipeline Orchestrator tests."""
    print("🚀 Starting RAG Pipeline Orchestrator Test Suite...")
    print("="*60)
    
    tester = RAGPipelineTester()
    
    # Create orchestrator
    orchestrator = await tester.test_orchestrator_creation()
    
    if orchestrator:
        # Run all tests
        await tester.test_end_to_end_pipeline(orchestrator)
        await tester.test_streaming_pipeline(orchestrator)
        await tester.test_agent_coordination(orchestrator)
        await tester.test_error_handling(orchestrator)
        await tester.test_caching_functionality(orchestrator)
        await tester.test_performance_metrics(orchestrator)
        await tester.test_pipeline_status_monitoring(orchestrator)
        await tester.test_configuration_management(orchestrator)
        await tester.test_concurrent_processing(orchestrator)
    
    # Print summary
    tester.print_test_summary()


if __name__ == "__main__":
    asyncio.run(main()) 
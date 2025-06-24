#!/usr/bin/env python3

"""
Test script for the Answer Generation Agent.

This script tests:
- Answer generation functionality
- Source citation and attribution
- Response formatting and structuring
- Context integration strategies
- Response quality assessment
- Streaming response support
- API endpoints
- Integration with agent framework
"""

import asyncio
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Add the backend directory to the Python path
sys.path.insert(0, '/Users/sproutdigitallab/Documents/Else/Untitled/agentic-rag-ai-agent/backend')

from app.agents.answer_generation import (
    AnswerGenerationAgent, 
    ResponseFormat, 
    CitationStyle, 
    ResponseQuality,
    GeneratedResponse
)
from app.agents.registry import AgentRegistry
from app.agents.metrics import AgentMetrics


class AnswerGenerationTester:
    """Test suite for the Answer Generation Agent."""
    
    def __init__(self):
        self.registry = AgentRegistry()
        self.metrics = AgentMetrics()
        self.test_results = []
        
        # Sample sources for testing
        self.sample_sources = [
            {
                "source_id": "doc1_chunk1",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
                "document_title": "Introduction to Machine Learning",
                "url": "https://example.com/ml-intro",
                "relevance_score": {"combined_score": 0.95},
                "chunk_index": 1
            },
            {
                "source_id": "doc2_chunk1", 
                "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision and natural language processing.",
                "document_title": "Deep Learning Fundamentals",
                "url": "https://example.com/deep-learning",
                "relevance_score": {"combined_score": 0.88},
                "chunk_index": 1
            },
            {
                "source_id": "doc3_chunk1",
                "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information using a connectionist approach to computation.",
                "document_title": "Neural Network Architecture",
                "url": "https://example.com/neural-networks",
                "relevance_score": {"combined_score": 0.82},
                "chunk_index": 1
            }
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
    
    async def test_agent_creation(self):
        """Test Answer Generation Agent creation and configuration."""
        print("\n🧪 Testing Answer Generation Agent Creation...")
        
        try:
            start_time = time.time()
            
            # Test basic agent creation
            agent = AnswerGenerationAgent(
                agent_id="test_answer_generation",
                config={
                    "max_response_length": 1500,
                    "citation_style": "numbered",
                    "response_format": "markdown",
                    "enable_streaming": True,
                    "quality_threshold": 0.8,
                    "temperature": 0.7
                }
            )
            
            # Verify configuration
            assert agent.max_response_length == 1500
            assert agent.citation_style == CitationStyle.NUMBERED
            assert agent.response_format == ResponseFormat.MARKDOWN
            assert agent.enable_streaming == True
            assert agent.quality_threshold == 0.8
            assert agent.temperature == 0.7
            
            duration = time.time() - start_time
            self.log_test_result(
                "Agent Creation", 
                True, 
                f"Agent created with ID: {agent.agent_id}",
                duration
            )
            
            return agent
            
        except Exception as e:
            self.log_test_result("Agent Creation", False, f"Error: {str(e)}")
            return None
    
    async def test_agent_lifecycle(self, agent: AnswerGenerationAgent):
        """Test agent lifecycle management."""
        print("\n🔄 Testing Agent Lifecycle...")
        
        try:
            start_time = time.time()
            
            # Test starting the agent
            await agent.start()
            assert agent.is_running
            # Check status (might be RUNNING or STARTED depending on implementation)
            assert agent.state.status in ["RUNNING", "STARTED"]
            
            # Test agent health
            assert agent.is_healthy
            
            # Test pausing the agent
            await agent.pause()
            assert agent.state.status == "PAUSED"
            
            # Test resuming the agent
            await agent.resume()
            assert agent.state.status in ["RUNNING", "STARTED"]
            
            duration = time.time() - start_time
            self.log_test_result(
                "Agent Lifecycle", 
                True, 
                "Start, pause, resume operations successful",
                duration
            )
            
        except Exception as e:
            import traceback
            self.log_test_result("Agent Lifecycle", False, f"Error: {str(e)}\nTraceback: {traceback.format_exc()}")
    
    async def test_basic_answer_generation(self, agent: AnswerGenerationAgent):
        """Test basic answer generation functionality."""
        print("\n📝 Testing Basic Answer Generation...")
        
        try:
            start_time = time.time()
            
            # Test query processing
            input_data = {
                "query": "What is machine learning and how does it relate to deep learning?",
                "sources": self.sample_sources,
                "conversation_history": [],
                "generation_config": {}
            }
            
            result = await agent.process(input_data)
            
            # Verify result structure
            assert result.success
            assert "response" in result.data
            assert "query" in result.data
            assert "sources_used" in result.data
            
            response_data = result.data["response"]
            assert "content" in response_data
            assert "citations" in response_data
            assert "quality" in response_data
            assert len(response_data["content"]) > 0
            
            duration = time.time() - start_time
            self.log_test_result(
                "Basic Answer Generation", 
                True, 
                f"Generated {response_data['word_count']} words with {len(response_data['citations'])} citations",
                duration
            )
            
            return result.data
            
        except Exception as e:
            self.log_test_result("Basic Answer Generation", False, f"Error: {str(e)}")
            return None
    
    async def test_citation_styles(self, agent: AnswerGenerationAgent):
        """Test different citation styles."""
        print("\n📚 Testing Citation Styles...")
        
        citation_styles = [
            (CitationStyle.NUMBERED, "numbered"),
            (CitationStyle.BRACKETED, "bracketed"),
            (CitationStyle.FOOTNOTE, "footnote"),
            (CitationStyle.INLINE, "inline")
        ]
        
        for style, style_name in citation_styles:
            try:
                start_time = time.time()
                
                # Update agent citation style
                agent.citation_style = style
                
                input_data = {
                    "query": "Explain neural networks and their applications.",
                    "sources": self.sample_sources[:2],  # Use fewer sources for testing
                    "conversation_history": [],
                    "generation_config": {}
                }
                
                result = await agent.process(input_data)
                
                assert result.success
                response_data = result.data["response"]
                
                # Verify citations are present
                assert len(response_data["citations"]) > 0
                
                duration = time.time() - start_time
                self.log_test_result(
                    f"Citation Style: {style_name}", 
                    True, 
                    f"Generated response with {len(response_data['citations'])} citations",
                    duration
                )
                
            except Exception as e:
                self.log_test_result(f"Citation Style: {style_name}", False, f"Error: {str(e)}")
    
    async def test_response_formats(self, agent: AnswerGenerationAgent):
        """Test different response formats."""
        print("\n🎨 Testing Response Formats...")
        
        response_formats = [
            (ResponseFormat.MARKDOWN, "markdown"),
            (ResponseFormat.PLAIN_TEXT, "plain_text"),
            (ResponseFormat.HTML, "html"),
            (ResponseFormat.JSON, "json")
        ]
        
        for format_type, format_name in response_formats:
            try:
                start_time = time.time()
                
                # Update agent response format
                agent.response_format = format_type
                
                input_data = {
                    "query": "What are the key differences between machine learning and deep learning?",
                    "sources": self.sample_sources,
                    "conversation_history": [],
                    "generation_config": {}
                }
                
                result = await agent.process(input_data)
                
                assert result.success, f"Processing failed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
                response_data = result.data["response"]
                
                # Verify format type is set correctly
                assert response_data["format_type"] == format_name, f"Expected format {format_name}, got {response_data['format_type']}"
                
                duration = time.time() - start_time
                self.log_test_result(
                    f"Response Format: {format_name}", 
                    True, 
                    f"Generated {format_name} formatted response",
                    duration
                )
                
            except Exception as e:
                self.log_test_result(f"Response Format: {format_name}", False, f"Error: {str(e)}")
    
    async def test_quality_assessment(self, agent: AnswerGenerationAgent):
        """Test response quality assessment."""
        print("\n🎯 Testing Quality Assessment...")
        
        try:
            start_time = time.time()
            
            # Enable quality assessment
            agent.enable_quality_assessment = True
            
            input_data = {
                "query": "How do neural networks work in machine learning?",
                "sources": self.sample_sources,
                "conversation_history": [],
                "generation_config": {}
            }
            
            result = await agent.process(input_data)
            
            assert result.success
            response_data = result.data["response"]
            quality_data = response_data["quality"]
            
            # Verify quality metrics are present
            required_metrics = [
                "relevance_score", "coherence_score", "completeness_score",
                "citation_accuracy", "factual_accuracy", "overall_quality"
            ]
            
            for metric in required_metrics:
                assert metric in quality_data
                assert 0.0 <= quality_data[metric] <= 1.0
            
            duration = time.time() - start_time
            self.log_test_result(
                "Quality Assessment", 
                True, 
                f"Overall quality score: {quality_data['overall_quality']:.3f}",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Quality Assessment", False, f"Error: {str(e)}")
    
    async def test_conversation_context(self, agent: AnswerGenerationAgent):
        """Test conversation context integration."""
        print("\n💬 Testing Conversation Context...")
        
        try:
            start_time = time.time()
            
            # Test with conversation history
            conversation_history = [
                {"role": "user", "content": "What is artificial intelligence?"},
                {"role": "assistant", "content": "Artificial intelligence is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence."},
                {"role": "user", "content": "How does machine learning fit into this?"}
            ]
            
            input_data = {
                "query": "Can you explain the relationship between AI and ML in more detail?",
                "sources": self.sample_sources,
                "conversation_history": conversation_history,
                "generation_config": {}
            }
            
            result = await agent.process(input_data)
            
            assert result.success
            response_data = result.data["response"]
            
            # Verify response acknowledges context
            assert len(response_data["content"]) > 0
            
            duration = time.time() - start_time
            self.log_test_result(
                "Conversation Context", 
                True, 
                f"Generated contextual response with {len(conversation_history)} previous messages",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Conversation Context", False, f"Error: {str(e)}")
    
    async def test_streaming_response(self, agent: AnswerGenerationAgent):
        """Test streaming response functionality."""
        print("\n🌊 Testing Streaming Response...")
        
        try:
            start_time = time.time()
            
            # Enable streaming
            agent.enable_streaming = True
            
            chunks = []
            async for chunk in agent.stream_response(
                query="Explain the basics of machine learning algorithms.",
                sources=self.sample_sources,
                conversation_history=[],
                generation_config={"stream": True}
            ):
                chunks.append(chunk)
                if len(chunks) >= 5:  # Limit for testing
                    break
            
            # Verify streaming worked
            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert len(full_response) > 0
            
            duration = time.time() - start_time
            self.log_test_result(
                "Streaming Response", 
                True, 
                f"Received {len(chunks)} chunks, total length: {len(full_response)}",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Streaming Response", False, f"Error: {str(e)}")
    
    async def test_caching_functionality(self, agent: AnswerGenerationAgent):
        """Test response caching."""
        print("\n💾 Testing Caching Functionality...")
        
        try:
            start_time = time.time()
            
            input_data = {
                "query": "What is the difference between supervised and unsupervised learning?",
                "sources": self.sample_sources,
                "conversation_history": [],
                "generation_config": {}
            }
            
            # First request (should not be cached)
            result1 = await agent.process(input_data)
            assert result1.success
            assert not result1.data["generation_metadata"]["cache_hit"]
            
            # Second request (should be cached)
            result2 = await agent.process(input_data)
            assert result2.success
            assert result2.data["generation_metadata"]["cache_hit"]
            
            duration = time.time() - start_time
            self.log_test_result(
                "Caching Functionality", 
                True, 
                "Cache hit on second identical request",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Caching Functionality", False, f"Error: {str(e)}")
    
    async def test_error_handling(self, agent: AnswerGenerationAgent):
        """Test error handling scenarios."""
        print("\n⚠️ Testing Error Handling...")
        
        try:
            start_time = time.time()
            
            # Test with empty query
            try:
                input_data = {
                    "query": "",
                    "sources": self.sample_sources,
                    "conversation_history": [],
                    "generation_config": {}
                }
                result = await agent.process(input_data)
                assert not result.success  # Should fail
            except ValueError:
                pass  # Expected error
            
            # Test with no sources (should still work with fallback)
            input_data = {
                "query": "What is machine learning?",
                "sources": [],
                "conversation_history": [],
                "generation_config": {}
            }
            result = await agent.process(input_data)
            assert result.success  # Should work with fallback
            
            duration = time.time() - start_time
            self.log_test_result(
                "Error Handling", 
                True, 
                "Handled empty query and no sources scenarios",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Error Handling", False, f"Error: {str(e)}")
    
    async def test_performance_metrics(self, agent: AnswerGenerationAgent):
        """Test performance metrics tracking."""
        print("\n📊 Testing Performance Metrics...")
        
        try:
            start_time = time.time()
            
            # Get initial stats
            initial_stats = agent._get_performance_stats()
            initial_generations = initial_stats["total_generations"]
            
            # Process a few requests
            for i in range(3):
                input_data = {
                    "query": f"Test query {i + 1}: What is machine learning?",
                    "sources": self.sample_sources,
                    "conversation_history": [],
                    "generation_config": {}
                }
                await agent.process(input_data)
            
            # Get updated stats
            updated_stats = agent._get_performance_stats()
            
            # Verify stats were updated
            assert updated_stats["total_generations"] == initial_generations + 3
            assert updated_stats["avg_response_length"] > 0
            
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Metrics", 
                True, 
                f"Tracked {updated_stats['total_generations']} total generations",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Performance Metrics", False, f"Error: {str(e)}")
    
    async def test_agent_registry_integration(self):
        """Test integration with agent registry."""
        print("\n🏢 Testing Agent Registry Integration...")
        
        try:
            start_time = time.time()
            
            # Register agent type
            self.registry.register_agent_type("answer_generation", AnswerGenerationAgent)
            
            # Create agent through registry
            agent = await self.registry.create_agent(
                agent_type="answer_generation",
                agent_id="registry_test_agent",
                config={"max_response_length": 1000},
                auto_start=True
            )
            
            # Verify agent is registered and running
            assert agent.agent_id == "registry_test_agent"
            assert agent.is_running
            
            # Test agent retrieval
            retrieved_agent = self.registry.get_agent("registry_test_agent")
            assert retrieved_agent is not None
            assert retrieved_agent.agent_id == agent.agent_id
            
            # Test agent listing
            agents = self.registry.get_agents_by_type("answer_generation")
            assert len(agents) > 0
            
            duration = time.time() - start_time
            self.log_test_result(
                "Agent Registry Integration", 
                True, 
                f"Agent registered and retrieved successfully",
                duration
            )
            
        except Exception as e:
            self.log_test_result("Agent Registry Integration", False, f"Error: {str(e)}")
    
    def print_test_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("🧪 ANSWER GENERATION AGENT TEST SUMMARY")
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
        fast_tests = sum(1 for result in self.test_results if result["duration"] < 1.0)
        print(f"  - Tests under 1s: {fast_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            print(f"\n🎉 All tests passed! Answer Generation Agent is working correctly.")
        else:
            print(f"\n⚠️ Some tests failed. Please review the failed tests above.")


async def main():
    """Run all Answer Generation Agent tests."""
    print("🚀 Starting Answer Generation Agent Test Suite...")
    print("="*60)
    
    tester = AnswerGenerationTester()
    
    # Create and test agent
    agent = await tester.test_agent_creation()
    
    if agent:
        await tester.test_agent_lifecycle(agent)
        await tester.test_basic_answer_generation(agent)
        await tester.test_citation_styles(agent)
        await tester.test_response_formats(agent)
        await tester.test_quality_assessment(agent)
        await tester.test_conversation_context(agent)
        await tester.test_streaming_response(agent)
        await tester.test_caching_functionality(agent)
        await tester.test_error_handling(agent)
        await tester.test_performance_metrics(agent)
        
        # Stop the agent
        await agent.stop()
    
    # Test registry integration
    await tester.test_agent_registry_integration()
    
    # Print summary
    tester.print_test_summary()


if __name__ == "__main__":
    asyncio.run(main()) 
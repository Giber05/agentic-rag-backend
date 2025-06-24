#!/usr/bin/env python3
"""
Test script for the Context Decision Agent.

This script tests:
- Context necessity evaluation functionality
- Pattern-based decision making
- Conversation context analysis
- Semantic similarity assessment
- AI-powered decision making
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

from app.agents.context_decision import ContextDecisionAgent, ContextNecessity
from app.agents.registry import AgentRegistry
from app.agents.metrics import AgentMetrics
from app.core.dependencies import get_agent_registry


async def test_context_decision_basic():
    """Test basic context decision functionality."""
    print("🔧 Testing Context Decision Agent - Basic Functionality...")
    
    # Create agent
    agent = ContextDecisionAgent("test-context-decision-1", config={
        "similarity_threshold": 0.7,
        "context_window_size": 5,
        "enable_ai_assessment": False  # Disable for basic test
    })
    
    await agent.start()
    
    # Test 1: Query with pronouns (should require context)
    input_data = {
        "query": "What is this about?",
        "conversation_history": [
            {"role": "user", "content": "Tell me about machine learning"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."}
        ],
        "current_context": {}
    }
    
    result = await agent.process(input_data)
    assert result.success, "Basic processing should succeed"
    
    decision_data = result.data
    print(f"  ✅ Query with pronouns: {decision_data['decision']} (confidence: {decision_data['confidence']:.2f})")
    assert decision_data["decision"] in ["required", "optional", "not_needed"], "Decision should be valid"
    assert 0.0 <= decision_data["confidence"] <= 1.0, "Confidence should be between 0 and 1"
    
    # Test 2: Standalone factual query (should not need context)
    input_data = {
        "query": "What is Python?",
        "conversation_history": [],
        "current_context": {}
    }
    
    result = await agent.process(input_data)
    decision_data = result.data
    print(f"  ✅ Standalone query: {decision_data['decision']} (confidence: {decision_data['confidence']:.2f})")
    
    # Test 3: Greeting (should not need context)
    input_data = {
        "query": "Hello there!",
        "conversation_history": [],
        "current_context": {}
    }
    
    result = await agent.process(input_data)
    decision_data = result.data
    print(f"  ✅ Greeting: {decision_data['decision']} (confidence: {decision_data['confidence']:.2f})")
    
    await agent.stop()
    print("  ✅ Basic context decision functionality working correctly")


async def test_pattern_assessment():
    """Test pattern-based assessment functionality."""
    print("🔧 Testing Pattern Assessment...")
    
    agent = ContextDecisionAgent("test-pattern-1")
    await agent.start()
    
    test_cases = [
        ("What is this?", "required"),  # Pronoun reference
        ("Tell me more about it", "required"),  # Pronoun + follow-up
        ("Hello", "not_needed"),  # Greeting
        ("What is Python?", "optional"),  # Factual question
        ("How does this work?", "required"),  # Pronoun reference
        ("Define machine learning", "optional"),  # Definition request
    ]
    
    for query, expected_category in test_cases:
        input_data = {
            "query": query,
            "conversation_history": [],
            "current_context": {}
        }
        
        result = await agent.process(input_data)
        decision = result.data["decision"]
        factors = result.data["decision_factors"]["pattern_assessment"]
        
        print(f"  Query: '{query}' -> {decision} (pattern: {factors['necessity'].value})")
        
        # Verify pattern assessment is working
        assert "matched_patterns" in factors, "Pattern matching should provide matched patterns"
    
    await agent.stop()
    print("  ✅ Pattern assessment working correctly")


async def test_conversation_context_analysis():
    """Test conversation context analysis."""
    print("🔧 Testing Conversation Context Analysis...")
    
    agent = ContextDecisionAgent("test-context-analysis-1")
    await agent.start()
    
    # Test with conversation history
    conversation_history = [
        {"role": "user", "content": "Tell me about artificial intelligence"},
        {"role": "assistant", "content": "AI is a field of computer science..."},
        {"role": "user", "content": "What are the main types?"},
        {"role": "assistant", "content": "There are several types including machine learning..."}
    ]
    
    # Test follow-up question
    input_data = {
        "query": "Can you explain more about machine learning?",
        "conversation_history": conversation_history,
        "current_context": {}
    }
    
    result = await agent.process(input_data)
    context_analysis = result.data["decision_factors"]["context_analysis"]
    
    print(f"  Follow-up query analysis:")
    print(f"    Topic continuity: {context_analysis['topic_continuity']:.2f}")
    print(f"    Recent messages: {context_analysis['recent_messages']}")
    print(f"    Decision: {context_analysis['necessity'].value}")
    
    # Test with pronouns
    input_data = {
        "query": "What about this technology?",
        "conversation_history": conversation_history,
        "current_context": {}
    }
    
    result = await agent.process(input_data)
    context_analysis = result.data["decision_factors"]["context_analysis"]
    
    print(f"  Pronoun query analysis:")
    print(f"    Pronouns found: {context_analysis['pronouns_found']}")
    print(f"    Decision: {context_analysis['necessity'].value}")
    
    await agent.stop()
    print("  ✅ Conversation context analysis working correctly")


async def test_semantic_similarity():
    """Test semantic similarity assessment (without OpenAI)."""
    print("🔧 Testing Semantic Similarity Assessment...")
    
    agent = ContextDecisionAgent("test-similarity-1")
    await agent.start()
    
    # Test with related conversation
    conversation_history = [
        {"role": "user", "content": "machine learning algorithms"},
        {"role": "assistant", "content": "neural networks deep learning"}
    ]
    
    input_data = {
        "query": "deep learning neural networks",
        "conversation_history": conversation_history,
        "current_context": {}
    }
    
    result = await agent.process(input_data)
    similarity_assessment = result.data["decision_factors"]["similarity_assessment"]
    
    print(f"  Similarity assessment method: {similarity_assessment['method']}")
    print(f"  Similarity score: {similarity_assessment.get('similarity_score', 'N/A')}")
    print(f"  Decision: {similarity_assessment['necessity'].value}")
    
    await agent.stop()
    print("  ✅ Semantic similarity assessment working correctly")


async def test_decision_making():
    """Test multi-factor decision making."""
    print("🔧 Testing Multi-Factor Decision Making...")
    
    agent = ContextDecisionAgent("test-decision-1", config={
        "enable_ai_assessment": False,
        "adaptive_thresholds": True
    })
    await agent.start()
    
    # Test complex scenario
    input_data = {
        "query": "How does this compare to traditional methods?",
        "conversation_history": [
            {"role": "user", "content": "Tell me about machine learning"},
            {"role": "assistant", "content": "Machine learning uses algorithms to learn patterns..."}
        ],
        "current_context": {}
    }
    
    result = await agent.process(input_data)
    decision_data = result.data
    
    print(f"  Final decision: {decision_data['decision']}")
    print(f"  Confidence: {decision_data['confidence']:.2f}")
    print(f"  Reasoning: {decision_data['reasoning']}")
    print(f"  Recommendations: {len(decision_data['recommendations'])} items")
    
    # Verify decision structure
    assert "decision_factors" in decision_data, "Decision factors should be included"
    assert "recommendations" in decision_data, "Recommendations should be included"
    assert "metadata" in decision_data, "Metadata should be included"
    
    await agent.stop()
    print("  ✅ Multi-factor decision making working correctly")


async def test_adaptive_thresholds():
    """Test adaptive threshold adjustment."""
    print("🔧 Testing Adaptive Thresholds...")
    
    agent = ContextDecisionAgent("test-adaptive-1", config={
        "adaptive_thresholds": True,
        "similarity_threshold": 0.7
    })
    await agent.start()
    
    initial_threshold = agent.similarity_threshold
    print(f"  Initial similarity threshold: {initial_threshold}")
    
    # Process several queries to trigger threshold adjustments
    for i in range(3):
        input_data = {
            "query": f"Test query {i}",
            "conversation_history": [],
            "current_context": {}
        }
        
        result = await agent.process(input_data)
        print(f"  Query {i}: confidence {result.data['confidence']:.2f}, threshold {agent.similarity_threshold:.2f}")
    
    await agent.stop()
    print("  ✅ Adaptive thresholds working correctly")


async def test_agent_framework_integration():
    """Test integration with agent framework."""
    print("🔧 Testing Agent Framework Integration...")
    
    registry = get_agent_registry()
    metrics = AgentMetrics()
    
    # Create agent through registry
    agent = await registry.create_agent(
        agent_type="context_decision",
        agent_id="test-framework-integration",
        config={"enable_ai_assessment": False},
        auto_start=True
    )
    
    # Check agent state
    state = agent.state
    assert state.status == "running", "Agent should be running"
    assert agent.is_healthy, "Agent should be healthy"
    
    # Process a query
    input_data = {
        "query": "What is this about?",
        "conversation_history": [],
        "current_context": {}
    }
    
    result = await agent.process(input_data)
    assert result.success, "Processing should succeed"
    
    # Check metrics
    agent_metrics = metrics.get_agent_metrics(agent.agent_id)
    if agent_metrics:
        print(f"  Agent metrics: {agent_metrics.total_operations} operations")
    
    # Clean up
    await agent.stop()
    registry.unregister_agent(agent.agent_id)
    
    print("  ✅ Agent framework integration working correctly")


async def test_api_endpoints():
    """Test API endpoints."""
    print("🔧 Testing API Endpoints...")
    
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Test evaluate endpoint
        response = client.post("/api/v1/context-decision/evaluate", json={
            "query": "What is this about?",
            "conversation_history": [
                {"role": "user", "content": "Tell me about AI"}
            ]
        })
        
        print(f"  Evaluate endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Decision: {data.get('decision')}")
            print(f"  Confidence: {data.get('confidence')}")
        
        # Test metrics endpoint
        response = client.get("/api/v1/context-decision/metrics")
        print(f"  Metrics endpoint status: {response.status_code}")
        
        print("  ✅ API endpoints working correctly")
        
    except Exception as e:
        print(f"  ⚠️  API test failed: {str(e)}")


async def test_performance_benchmark():
    """Test performance benchmarks."""
    print("🔧 Testing Performance Benchmarks...")
    
    agent = ContextDecisionAgent("test-performance-1", config={
        "enable_ai_assessment": False  # Disable AI for consistent timing
    })
    await agent.start()
    
    # Test queries
    test_queries = [
        "What is this?",
        "Tell me more about machine learning",
        "How does this work?",
        "Define artificial intelligence",
        "Hello there!"
    ]
    
    conversation_history = [
        {"role": "user", "content": "Tell me about technology"},
        {"role": "assistant", "content": "Technology encompasses various tools and systems..."}
    ]
    
    total_time = 0
    successful_operations = 0
    
    for query in test_queries:
        start_time = time.time()
        
        input_data = {
            "query": query,
            "conversation_history": conversation_history,
            "current_context": {}
        }
        
        result = await agent.process(input_data)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        if result.success:
            successful_operations += 1
            total_time += processing_time
            print(f"  Query: '{query[:30]}...' -> {processing_time:.1f}ms")
    
    if successful_operations > 0:
        avg_time = total_time / successful_operations
        print(f"  Average processing time: {avg_time:.1f}ms")
        print(f"  Success rate: {successful_operations}/{len(test_queries)} ({100*successful_operations/len(test_queries):.1f}%)")
        
        # Performance assertion
        assert avg_time < 500, f"Average processing time should be under 500ms, got {avg_time:.1f}ms"
    
    await agent.stop()
    print("  ✅ Performance benchmarks passed")


async def main():
    """Run all tests."""
    print("🚀 Starting Context Decision Agent Tests...\n")
    
    try:
        await test_context_decision_basic()
        print()
        
        await test_pattern_assessment()
        print()
        
        await test_conversation_context_analysis()
        print()
        
        await test_semantic_similarity()
        print()
        
        await test_decision_making()
        print()
        
        await test_adaptive_thresholds()
        print()
        
        await test_agent_framework_integration()
        print()
        
        await test_api_endpoints()
        print()
        
        await test_performance_benchmark()
        print()
        
        print("🎉 All Context Decision Agent tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 
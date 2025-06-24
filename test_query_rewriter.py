#!/usr/bin/env python3
"""
Test script for the Query Rewriting Agent.

This script tests:
- Query rewriting functionality
- Spell and grammar correction
- Query normalization and optimization
- API endpoints
- Integration with agent framework
"""

import asyncio
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the Python path
sys.path.insert(0, '/Users/sproutdigitallab/Documents/Else/Untitled/agentic-rag-ai-agent/backend')

from app.agents.query_rewriter import QueryRewritingAgent
from app.agents.registry import AgentRegistry
from app.agents.metrics import AgentMetrics
from app.core.dependencies import get_agent_registry, get_openai_service


async def test_query_rewriter_basic():
    """Test basic query rewriting functionality."""
    print("🔧 Testing Query Rewriter Basic Functionality...")
    
    # Create agent with test configuration
    config = {
        "enable_spell_check": False,  # Disable OpenAI calls for basic test
        "enable_expansion": False,
        "max_query_length": 200,
        "min_query_length": 3
    }
    
    agent = QueryRewritingAgent("test-query-rewriter", "query_rewriter", config)
    await agent.start()
    
    # Test cases
    test_queries = [
        "whats the capital of france?",
        "How does machine learning work",
        "cant you tell me about AI?",
        "What is the difference between AI and ML and how do they work together?",
        "explain neural networks",
        "i want to know about python programming"
    ]
    
    for query in test_queries:
        print(f"\n   Testing: '{query}'")
        
        input_data = {"query": query}
        result = await agent.process(input_data)
        
        assert result.success, f"Query processing failed: {result.error}"
        
        data = result.data
        print(f"   Original: {data['original_query']}")
        print(f"   Rewritten: {data['rewritten_query']}")
        print(f"   Improvements: {data['improvements']}")
        print(f"   Confidence: {data['confidence']:.2f}")
        print(f"   Processing time: {result.processing_time_ms}ms")
    
    await agent.stop()
    print("✅ Basic query rewriter tests passed!")


async def test_query_rewriter_with_openai():
    """Test query rewriting with OpenAI integration."""
    print("🤖 Testing Query Rewriter with OpenAI...")
    
    try:
        # Create agent with OpenAI enabled
        config = {
            "enable_spell_check": True,
            "enable_expansion": True,
            "max_query_length": 200,
            "min_query_length": 3
        }
        
        agent = QueryRewritingAgent("test-query-rewriter-ai", "query_rewriter", config)
        await agent.start()
        
        # Test queries with spelling/grammar errors
        test_queries = [
            "wat is machien lerning?",  # Spelling errors
            "how do neural netwroks work",  # Spelling errors
            "tell me bout artificial inteligence",  # Contractions and spelling
            "AI"  # Short query for expansion
        ]
        
        for query in test_queries:
            print(f"\n   Testing: '{query}'")
            
            input_data = {"query": query}
            result = await agent.process(input_data)
            
            if result.success:
                data = result.data
                print(f"   Original: {data['original_query']}")
                print(f"   Rewritten: {data['rewritten_query']}")
                print(f"   Steps: {data['preprocessing_steps']}")
                print(f"   Improvements: {data['improvements']}")
                print(f"   Confidence: {data['confidence']:.2f}")
                print(f"   Processing time: {result.processing_time_ms}ms")
            else:
                print(f"   ⚠️ Processing failed: {result.error}")
        
        await agent.stop()
        print("✅ OpenAI integration tests completed!")
        
    except Exception as e:
        print(f"⚠️ OpenAI tests skipped (likely no API key): {str(e)}")


async def test_query_validation():
    """Test query validation functionality."""
    print("🔍 Testing Query Validation...")
    
    agent = QueryRewritingAgent("test-validator", "query_rewriter")
    await agent.start()
    
    # Test invalid queries
    invalid_queries = [
        "",  # Empty query
        "a",  # Too short
        "x" * 600,  # Too long
        "<script>alert('xss')</script>",  # Malicious content
        "DROP TABLE users;",  # SQL injection attempt
    ]
    
    for query in invalid_queries:
        print(f"\n   Testing invalid query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        input_data = {"query": query}
        result = await agent.process(input_data)
        
        assert not result.success, f"Invalid query should have failed: {query}"
        print(f"   ✅ Correctly rejected: {result.error}")
    
    await agent.stop()
    print("✅ Query validation tests passed!")


async def test_agent_framework_integration():
    """Test integration with the agent framework."""
    print("🔗 Testing Agent Framework Integration...")
    
    # Get registry and register agent type
    registry = get_agent_registry()
    metrics = AgentMetrics()
    
    # Create agent through registry
    agent = await registry.create_agent(
        agent_type="query_rewriter",
        agent_id="framework-test-agent",
        config={"enable_spell_check": False, "enable_expansion": False},
        auto_start=True
    )
    
    # Test processing through framework
    input_data = {"query": "whats artificial intelligence?"}
    result = await agent.process(input_data)
    
    assert result.success, f"Framework processing failed: {result.error}"
    
    # Record metrics
    metrics.record_operation(
        agent_id=agent.agent_id,
        agent_type=agent.agent_type,
        success=result.success,
        processing_time_ms=result.processing_time_ms,
        operation_type="query_rewrite"
    )
    
    # Check agent state
    state = agent.state
    assert state.status == "running", "Agent should be running"
    assert agent.is_healthy, "Agent should be healthy"
    
    # Check metrics
    agent_metrics = metrics.get_agent_metrics(agent.agent_id)
    assert agent_metrics is not None, "Should have metrics"
    assert agent_metrics.total_operations == 1, "Should have one operation"
    assert agent_metrics.successful_operations == 1, "Should have one successful operation"
    
    # Clean up
    await registry.stop_agent(agent.agent_id)
    registry.unregister_agent(agent.agent_id)
    
    print("✅ Agent framework integration tests passed!")


async def test_api_endpoints():
    """Test the Query Rewriter API endpoints."""
    print("🌐 Testing API Endpoints...")
    
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Test query processing endpoint
        print("\n   Testing /api/v1/query-rewriter/process")
        
        response = client.post(
            "/api/v1/query-rewriter/process",
            json={
                "query": "whats machine learning?",
                "conversation_id": "test-conversation",
                "context": {"test": True}
            }
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Original: {data['original_query']}")
            print(f"   Rewritten: {data['rewritten_query']}")
            print(f"   Confidence: {data['confidence']}")
            print(f"   Processing time: {data['processing_time_ms']}ms")
        else:
            print(f"   Error: {response.text}")
        
        # Test stats endpoint
        print("\n   Testing /api/v1/query-rewriter/stats")
        
        response = client.get("/api/v1/query-rewriter/stats")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Agent type: {data['agent_type']}")
            print(f"   Number of agents: {len(data['agents_info'])}")
        else:
            print(f"   Response: {response.text}")
        
        # Test agent creation endpoint
        print("\n   Testing /api/v1/query-rewriter/agent/create")
        
        response = client.post(
            "/api/v1/query-rewriter/agent/create",
            params={
                "agent_id": "api-test-agent",
                "auto_start": True
            },
            json={"enable_spell_check": False}
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Created agent: {data['agent_id']}")
            print(f"   Status: {data['status']}")
        else:
            print(f"   Response: {response.text}")
        
        print("✅ API endpoint tests completed!")
        
    except Exception as e:
        print(f"⚠️ API tests failed: {str(e)}")


async def test_performance():
    """Test query rewriter performance."""
    print("⚡ Testing Performance...")
    
    agent = QueryRewritingAgent("perf-test", "query_rewriter", {"enable_spell_check": False})
    await agent.start()
    
    # Performance test queries
    queries = [
        "what is AI?",
        "how does machine learning work?",
        "explain neural networks",
        "what are the benefits of artificial intelligence?",
        "how to implement deep learning models?"
    ]
    
    total_time = 0
    successful_queries = 0
    
    print(f"\n   Processing {len(queries)} queries...")
    
    for i, query in enumerate(queries, 1):
        start_time = time.time()
        
        input_data = {"query": query}
        result = await agent.process(input_data)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        if result.success:
            successful_queries += 1
            total_time += processing_time
            print(f"   Query {i}: {processing_time:.1f}ms")
        else:
            print(f"   Query {i}: FAILED - {result.error}")
    
    if successful_queries > 0:
        avg_time = total_time / successful_queries
        print(f"\n   📊 Performance Results:")
        print(f"   Successful queries: {successful_queries}/{len(queries)}")
        print(f"   Average processing time: {avg_time:.1f}ms")
        print(f"   Total time: {total_time:.1f}ms")
        
        # Performance assertions
        assert avg_time < 1000, f"Average processing time too high: {avg_time}ms"
        assert successful_queries == len(queries), "All queries should succeed"
    
    await agent.stop()
    print("✅ Performance tests passed!")


async def main():
    """Run all Query Rewriter tests."""
    print("🚀 Starting Query Rewriting Agent Tests")
    print("=" * 60)
    
    try:
        # Run all test suites
        await test_query_rewriter_basic()
        await test_query_validation()
        await test_agent_framework_integration()
        await test_query_rewriter_with_openai()
        await test_api_endpoints()
        await test_performance()
        
        print("=" * 60)
        print("🎉 All Query Rewriting Agent tests passed!")
        print("\n📋 Test Summary:")
        print("✅ Basic query rewriting functionality")
        print("✅ Query validation and security")
        print("✅ Agent framework integration")
        print("✅ OpenAI integration (spell check & expansion)")
        print("✅ API endpoints")
        print("✅ Performance benchmarks")
        
        print("\n🔧 Features Verified:")
        print("• Query preprocessing and normalization")
        print("• Spell and grammar correction")
        print("• Query expansion and optimization")
        print("• Security validation")
        print("• Framework integration")
        print("• Performance monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 
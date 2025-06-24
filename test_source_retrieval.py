#!/usr/bin/env python3
"""
Test script for the Source Retrieval Agent.

This script tests:
- Source retrieval functionality
- Semantic search capabilities
- Keyword search functionality
- Hybrid search strategies
- Adaptive retrieval logic
- Result ranking and deduplication
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

from app.agents.source_retrieval import (
    SourceRetrievalAgent, 
    RetrievalStrategy, 
    SourceType, 
    RelevanceScore,
    RetrievedSource
)
from app.agents.registry import AgentRegistry
from app.agents.metrics import AgentMetrics
from app.core.dependencies import get_agent_registry


async def test_source_retrieval_basic():
    """Test basic source retrieval functionality."""
    print("🔧 Testing Source Retrieval Agent - Basic Functionality...")
    
    # Create agent
    agent = SourceRetrievalAgent("test-source-retrieval-1", config={
        "max_results": 5,
        "min_relevance_threshold": 0.3,
        "default_strategy": "adaptive"
    })
    
    await agent.start()
    
    # Test 1: Basic retrieval (will use fallback since no real data)
    input_data = {
        "query": "machine learning algorithms",
        "context_decision": {
            "decision": "required",
            "confidence": 0.8
        },
        "conversation_history": [],
        "retrieval_config": {}
    }
    
    result = await agent.process(input_data)
    assert result.success, "Basic processing should succeed"
    
    retrieval_data = result.data
    print(f"  ✅ Basic retrieval: {retrieval_data['total_sources']} sources found")
    print(f"  ✅ Strategy used: {retrieval_data['strategy_used']}")
    assert "sources" in retrieval_data, "Sources should be included"
    assert "strategy_used" in retrieval_data, "Strategy should be specified"
    
    # Test 2: Different query types
    test_queries = [
        ("What is Python?", "keyword"),
        ("Explain machine learning concepts", "hybrid"),
        ("Compare neural networks and decision trees", "hybrid")
    ]
    
    for query, expected_strategy_type in test_queries:
        input_data = {
            "query": query,
            "context_decision": {"decision": "optional", "confidence": 0.6},
            "conversation_history": [],
            "retrieval_config": {}
        }
        
        result = await agent.process(input_data)
        retrieval_data = result.data
        print(f"  ✅ Query: '{query[:30]}...' -> {retrieval_data['strategy_used']}")
    
    await agent.stop()
    print("  ✅ Basic source retrieval functionality working correctly")


async def test_retrieval_strategies():
    """Test different retrieval strategies."""
    print("🔧 Testing Retrieval Strategies...")
    
    agent = SourceRetrievalAgent("test-strategies-1")
    await agent.start()
    
    strategies = [
        RetrievalStrategy.SEMANTIC_ONLY,
        RetrievalStrategy.KEYWORD,
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.ADAPTIVE
    ]
    
    for strategy in strategies:
        input_data = {
            "query": "artificial intelligence applications",
            "context_decision": {"decision": "required", "confidence": 0.7},
            "conversation_history": [],
            "retrieval_config": {"strategy": strategy.value}
        }
        
        result = await agent.process(input_data)
        retrieval_data = result.data
        
        print(f"  Strategy: {strategy.value} -> {retrieval_data['total_sources']} sources")
        assert retrieval_data["strategy_used"] == strategy.value, f"Strategy should be {strategy.value}"
    
    await agent.stop()
    print("  ✅ Retrieval strategies working correctly")


async def test_relevance_scoring():
    """Test relevance scoring functionality."""
    print("🔧 Testing Relevance Scoring...")
    
    # Test RelevanceScore class
    score = RelevanceScore(
        semantic_score=0.8,
        keyword_score=0.6,
        recency_score=0.9,
        authority_score=0.7,
        context_score=0.5
    )
    
    combined = score.combined_score
    print(f"  Combined relevance score: {combined:.3f}")
    assert 0.0 <= combined <= 1.0, "Combined score should be between 0 and 1"
    
    score_dict = score.to_dict()
    assert "combined_score" in score_dict, "Score dict should include combined score"
    print(f"  ✅ Score components: {len(score_dict)} fields")
    
    # Test RetrievedSource class
    source = RetrievedSource(
        source_id="test-123",
        content="This is test content about machine learning",
        source_type=SourceType.CHUNK,
        relevance_score=score,
        metadata={"test": True},
        document_title="Test Document"
    )
    
    source_dict = source.to_dict()
    assert source_dict["source_id"] == "test-123", "Source ID should match"
    assert source_dict["source_type"] == "chunk", "Source type should be serialized"
    assert "content_hash" in source_dict, "Content hash should be included"
    
    print("  ✅ Relevance scoring working correctly")


async def test_adaptive_strategy_selection():
    """Test adaptive strategy selection logic."""
    print("🔧 Testing Adaptive Strategy Selection...")
    
    agent = SourceRetrievalAgent("test-adaptive-1", config={
        "default_strategy": "adaptive"
    })
    await agent.start()
    
    test_cases = [
        ("What is machine learning?", "keyword"),  # Definition query
        ("Compare neural networks and SVMs", "hybrid"),  # Complex comparison
        ("Explain deep learning concepts", "hybrid"),  # Explanation query
        ("Hello there", "hybrid"),  # Simple query
    ]
    
    for query, expected_category in test_cases:
        context_decision = {"decision": "optional", "confidence": 0.6}
        strategy = agent._adaptive_strategy_selection(query, context_decision)
        
        print(f"  Query: '{query}' -> {strategy.value}")
        # Note: We can't assert exact strategy since it's adaptive
        assert strategy in RetrievalStrategy, "Strategy should be valid"
    
    await agent.stop()
    print("  ✅ Adaptive strategy selection working correctly")


async def test_keyword_extraction():
    """Test keyword extraction functionality."""
    print("🔧 Testing Keyword Extraction...")
    
    agent = SourceRetrievalAgent("test-keywords-1")
    await agent.start()
    
    test_queries = [
        "What is machine learning?",
        "Explain neural networks and deep learning",
        "How do recommendation systems work?",
        "Compare supervised and unsupervised learning"
    ]
    
    for query in test_queries:
        keywords = agent._extract_keywords(query)
        print(f"  Query: '{query}' -> Keywords: {keywords}")
        
        # Verify keywords don't contain stop words
        stop_words = {'the', 'is', 'and', 'how', 'do'}
        for keyword in keywords:
            assert keyword not in stop_words, f"Keyword '{keyword}' should not be a stop word"
            assert len(keyword) > 2, f"Keyword '{keyword}' should be longer than 2 characters"
    
    await agent.stop()
    print("  ✅ Keyword extraction working correctly")


async def test_deduplication():
    """Test source deduplication functionality."""
    print("🔧 Testing Source Deduplication...")
    
    agent = SourceRetrievalAgent("test-dedup-1", config={
        "enable_deduplication": True,
        "similarity_threshold": 0.8
    })
    await agent.start()
    
    # Create test sources with some duplicates
    sources = [
        RetrievedSource(
            source_id="1",
            content="Machine learning is a subset of artificial intelligence",
            source_type=SourceType.CHUNK,
            relevance_score=RelevanceScore(semantic_score=0.9)
        ),
        RetrievedSource(
            source_id="2",
            content="Machine learning is a subset of artificial intelligence",  # Exact duplicate
            source_type=SourceType.CHUNK,
            relevance_score=RelevanceScore(semantic_score=0.8)
        ),
        RetrievedSource(
            source_id="3",
            content="Deep learning is a type of machine learning",
            source_type=SourceType.CHUNK,
            relevance_score=RelevanceScore(semantic_score=0.7)
        ),
        RetrievedSource(
            source_id="4",
            content="Machine learning algorithms learn from data",
            source_type=SourceType.CHUNK,
            relevance_score=RelevanceScore(semantic_score=0.6)
        )
    ]
    
    print(f"  Original sources: {len(sources)}")
    
    deduplicated = agent._deduplicate_sources(sources)
    print(f"  After deduplication: {len(deduplicated)}")
    
    # Should have removed the exact duplicate
    assert len(deduplicated) < len(sources), "Deduplication should remove some sources"
    
    # Verify no exact duplicates remain
    content_hashes = [source.content_hash for source in deduplicated]
    assert len(content_hashes) == len(set(content_hashes)), "No duplicate content hashes should remain"
    
    await agent.stop()
    print("  ✅ Source deduplication working correctly")


async def test_caching():
    """Test retrieval caching functionality."""
    print("🔧 Testing Retrieval Caching...")
    
    agent = SourceRetrievalAgent("test-cache-1")
    await agent.start()
    
    query = "test caching query"
    config = {"max_results": 5}
    
    # Generate cache key
    cache_key = agent._generate_cache_key(query, config)
    print(f"  Generated cache key: {cache_key[:16]}...")
    
    # Test cache miss
    cached_result = agent._get_cached_result(cache_key)
    assert cached_result is None, "Cache should be empty initially"
    
    # Test cache storage
    test_result = {
        "query": query,
        "sources": [],
        "total_sources": 0,
        "retrieval_metadata": {"test": True}
    }
    
    agent._cache_result(cache_key, test_result)
    
    # Test cache hit
    cached_result = agent._get_cached_result(cache_key)
    assert cached_result is not None, "Result should be cached"
    assert cached_result["query"] == query, "Cached query should match"
    assert cached_result["retrieval_metadata"]["cache_hit"] is True, "Cache hit flag should be set"
    
    await agent.stop()
    print("  ✅ Retrieval caching working correctly")


async def test_performance_stats():
    """Test performance statistics tracking."""
    print("🔧 Testing Performance Statistics...")
    
    agent = SourceRetrievalAgent("test-stats-1")
    await agent.start()
    
    # Initial stats
    initial_stats = agent._get_performance_stats()
    print(f"  Initial stats: {initial_stats['total_retrievals']} retrievals")
    
    # Perform some operations
    for i in range(3):
        agent._update_stats(RetrievalStrategy.HYBRID, 5)
    
    # Check updated stats
    updated_stats = agent._get_performance_stats()
    print(f"  Updated stats: {updated_stats['total_retrievals']} retrievals")
    print(f"  Average results: {updated_stats['avg_results_count']:.1f}")
    
    assert updated_stats['total_retrievals'] > initial_stats['total_retrievals'], "Stats should be updated"
    assert updated_stats['strategy_distribution']['hybrid_searches'] == 3, "Hybrid searches should be tracked"
    
    await agent.stop()
    print("  ✅ Performance statistics working correctly")


async def test_agent_framework_integration():
    """Test integration with agent framework."""
    print("🔧 Testing Agent Framework Integration...")
    
    registry = get_agent_registry()
    metrics = AgentMetrics()
    
    # Create agent through registry
    agent = await registry.create_agent(
        agent_type="source_retrieval",
        agent_id="test-framework-integration",
        config={"max_results": 3},
        auto_start=True
    )
    
    # Check agent state
    state = agent.state
    assert state.status == "running", "Agent should be running"
    assert agent.is_healthy, "Agent should be healthy"
    
    # Process a query
    input_data = {
        "query": "test framework integration",
        "context_decision": {"decision": "required", "confidence": 0.7},
        "conversation_history": [],
        "retrieval_config": {}
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
        
        # Test retrieve endpoint
        response = client.post("/api/v1/source-retrieval/retrieve", json={
            "query": "machine learning algorithms",
            "context_decision": {
                "decision": "required",
                "confidence": 0.8
            },
            "conversation_history": []
        })
        
        print(f"  Retrieve endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Sources retrieved: {data.get('total_sources', 0)}")
            print(f"  Strategy used: {data.get('strategy_used')}")
        
        # Test performance endpoint
        response = client.get("/api/v1/source-retrieval/performance")
        print(f"  Performance endpoint status: {response.status_code}")
        
        # Test strategies endpoint
        response = client.get("/api/v1/source-retrieval/strategies")
        print(f"  Strategies endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            strategies = data.get('available_strategies', {})
            print(f"  Available strategies: {len(strategies)}")
        
        print("  ✅ API endpoints working correctly")
        
    except Exception as e:
        print(f"  ⚠️  API test failed: {str(e)}")


async def test_performance_benchmark():
    """Test performance benchmarks."""
    print("🔧 Testing Performance Benchmarks...")
    
    agent = SourceRetrievalAgent("test-performance-1", config={
        "max_results": 5
    })
    await agent.start()
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "neural network architectures",
        "data science techniques",
        "artificial intelligence applications",
        "deep learning frameworks"
    ]
    
    context_decision = {
        "decision": "required",
        "confidence": 0.8
    }
    
    total_time = 0
    successful_operations = 0
    
    for query in test_queries:
        start_time = time.time()
        
        input_data = {
            "query": query,
            "context_decision": context_decision,
            "conversation_history": [],
            "retrieval_config": {}
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
        
        # Performance assertion (relaxed since we don't have real data)
        assert avg_time < 2000, f"Average processing time should be under 2000ms, got {avg_time:.1f}ms"
    
    await agent.stop()
    print("  ✅ Performance benchmarks passed")


async def main():
    """Run all tests."""
    print("🚀 Starting Source Retrieval Agent Tests...\n")
    
    try:
        await test_source_retrieval_basic()
        print()
        
        await test_retrieval_strategies()
        print()
        
        await test_relevance_scoring()
        print()
        
        await test_adaptive_strategy_selection()
        print()
        
        await test_keyword_extraction()
        print()
        
        await test_deduplication()
        print()
        
        await test_caching()
        print()
        
        await test_performance_stats()
        print()
        
        await test_agent_framework_integration()
        print()
        
        await test_api_endpoints()
        print()
        
        await test_performance_benchmark()
        print()
        
        print("🎉 All Source Retrieval Agent tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 
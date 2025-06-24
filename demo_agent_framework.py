#!/usr/bin/env python3
"""
Demonstration script for the Agent Framework Foundation.

This script shows how to:
1. Create and register agents
2. Execute the RAG pipeline
3. Monitor agent performance
4. Use the API endpoints
"""

import asyncio
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the Python path
sys.path.insert(0, '/Users/sproutdigitallab/Documents/Else/Untitled/agentic-rag-ai-agent/backend')

from app.agents.registry import AgentRegistry
from app.agents.coordinator import AgentCoordinator
from app.agents.metrics import AgentMetrics
from app.agents.base import BaseAgent, AgentResult


class DemoQueryRewriterAgent(BaseAgent):
    """Demo Query Rewriter Agent."""
    
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Rewrite and optimize the query."""
        query = input_data.get("query", "")
        
        # Simulate query rewriting
        await asyncio.sleep(0.1)
        
        rewritten_query = f"optimized: {query.strip().lower()}"
        
        return {
            "rewritten_query": rewritten_query,
            "original_query": query,
            "improvements": ["lowercased", "optimized"],
            "confidence": 0.95
        }


class DemoContextDecisionAgent(BaseAgent):
    """Demo Context Decision Agent."""
    
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Decide if additional context is needed."""
        query = input_data.get("query", "")
        
        # Simulate decision making
        await asyncio.sleep(0.05)
        
        # Simple heuristic: need context if query is longer than 10 characters
        needs_context = len(query) > 10
        
        return {
            "needs_context": needs_context,
            "confidence": 0.85,
            "reasoning": f"Query length: {len(query)} characters",
            "decision": "retrieve" if needs_context else "skip"
        }


class DemoSourceRetrievalAgent(BaseAgent):
    """Demo Source Retrieval Agent."""
    
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Retrieve relevant sources."""
        query = input_data.get("query", "")
        
        # Simulate source retrieval
        await asyncio.sleep(0.2)
        
        # Mock retrieved sources
        sources = [
            {
                "id": "doc_1",
                "title": "Relevant Document 1",
                "content": f"This document contains information about {query}",
                "relevance_score": 0.92
            },
            {
                "id": "doc_2", 
                "title": "Relevant Document 2",
                "content": f"Additional context for {query}",
                "relevance_score": 0.78
            }
        ]
        
        return {
            "sources": sources,
            "total_found": len(sources),
            "search_query": query,
            "retrieval_time_ms": 200
        }


class DemoAnswerGenerationAgent(BaseAgent):
    """Demo Answer Generation Agent."""
    
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate answer with sources."""
        query = input_data.get("query", "")
        previous_results = input_data.get("previous_results", {})
        
        # Simulate answer generation
        await asyncio.sleep(0.3)
        
        # Get sources from previous step
        sources = previous_results.get("source_retrieval", {}).get("sources", [])
        
        answer = f"Based on the available sources, here's what I found about '{query}': "
        answer += "The retrieved documents provide relevant information. "
        
        if sources:
            answer += f"I found {len(sources)} relevant sources that support this answer."
        
        return {
            "answer": answer,
            "sources_used": [s["id"] for s in sources],
            "confidence": 0.88,
            "word_count": len(answer.split())
        }


class DemoValidationAgent(BaseAgent):
    """Demo Validation & Refinement Agent."""
    
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Validate and refine the answer."""
        previous_results = input_data.get("previous_results", {})
        
        # Simulate validation
        await asyncio.sleep(0.1)
        
        answer_data = previous_results.get("answer_generation", {})
        answer = answer_data.get("answer", "")
        confidence = answer_data.get("confidence", 0.0)
        
        # Simple validation: check if answer is long enough
        is_valid = len(answer) > 50
        quality_score = confidence * (1.0 if is_valid else 0.5)
        
        return {
            "is_valid": is_valid,
            "quality_score": quality_score,
            "validation_checks": ["length_check", "confidence_check"],
            "refinement_suggestions": [] if is_valid else ["expand_answer"],
            "final_answer": answer
        }


async def demo_agent_framework():
    """Demonstrate the agent framework capabilities."""
    print("🚀 Agent Framework Foundation Demo")
    print("=" * 50)
    
    # 1. Initialize components
    print("\n1️⃣ Initializing Framework Components...")
    registry = AgentRegistry()
    metrics = AgentMetrics()
    coordinator = AgentCoordinator(registry)
    
    # 2. Register agent types
    print("\n2️⃣ Registering Agent Types...")
    agent_types = {
        "query_rewriter": DemoQueryRewriterAgent,
        "context_decision": DemoContextDecisionAgent,
        "source_retrieval": DemoSourceRetrievalAgent,
        "answer_generation": DemoAnswerGenerationAgent,
        "validation_refinement": DemoValidationAgent
    }
    
    for agent_type, agent_class in agent_types.items():
        registry.register_agent_type(agent_type, agent_class)
        print(f"   ✅ Registered {agent_type}")
    
    # 3. Create and start agents
    print("\n3️⃣ Creating and Starting Agents...")
    agents = {}
    for agent_type in agent_types.keys():
        agent = await registry.create_agent(
            agent_type=agent_type,
            agent_id=f"demo_{agent_type}",
            auto_start=True
        )
        agents[agent_type] = agent
        print(f"   ✅ Started {agent.agent_id}")
    
    # 4. Show registry stats
    print("\n4️⃣ Registry Statistics:")
    stats = registry.get_registry_stats()
    print(f"   📊 Total agents: {stats['total_agents']}")
    print(f"   📊 Registered types: {stats['registered_types']}")
    print(f"   📊 Healthy agents: {stats['healthy_agents']}")
    print(f"   📊 Running agents: {stats['running_agents']}")
    
    # 5. Execute RAG pipeline
    print("\n5️⃣ Executing RAG Pipeline...")
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain neural networks"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        execution = await coordinator.execute_pipeline(
            query=query,
            conversation_id=f"demo_conversation_{i}",
            context={"demo": True, "query_number": i}
        )
        
        print(f"   📊 Status: {execution.status}")
        print(f"   📊 Duration: {execution.duration_ms}ms")
        print(f"   📊 Steps completed: {len(execution.step_results)}")
        
        # Record metrics for demonstration
        for step_name, result in execution.step_results.items():
            metrics.record_operation(
                agent_id=result.agent_id,
                agent_type=result.agent_type,
                success=result.success,
                processing_time_ms=result.processing_time_ms,
                operation_type=step_name
            )
        
        # Show final answer if available
        if execution.status == "completed":
            validation_result = execution.step_results.get("validation_refinement")
            if validation_result and validation_result.success:
                final_answer = validation_result.data.get("final_answer", "")
                print(f"   💬 Answer: {final_answer[:100]}...")
    
    # 6. Show coordinator stats
    print("\n6️⃣ Coordinator Statistics:")
    coord_stats = coordinator.get_coordinator_stats()
    print(f"   📊 Total executions: {coord_stats['total_executions']}")
    print(f"   📊 Completed: {coord_stats['completed_executions']}")
    print(f"   📊 Failed: {coord_stats['failed_executions']}")
    print(f"   📊 Success rate: {coord_stats['success_rate']:.2%}")
    print(f"   📊 Average duration: {coord_stats['average_duration_ms']:.1f}ms")
    
    # 7. Show agent metrics
    print("\n7️⃣ Agent Performance Metrics:")
    system_overview = metrics.get_system_overview()
    print(f"   📊 Total operations: {system_overview['total_operations']}")
    print(f"   📊 System success rate: {system_overview['system_success_rate']:.2%}")
    print(f"   📊 Average processing time: {system_overview['average_processing_time_ms']:.1f}ms")
    
    for agent_type in agent_types.keys():
        agent_id = f"demo_{agent_type}"
        agent_metrics = metrics.get_agent_metrics(agent_id)
        if agent_metrics:
            print(f"   📊 {agent_type}: {agent_metrics.total_operations} ops, "
                  f"{agent_metrics.success_rate:.2%} success, "
                  f"{agent_metrics.average_processing_time_ms:.1f}ms avg")
    
    # 8. Health check
    print("\n8️⃣ Health Check:")
    health = await coordinator.health_check()
    print(f"   🏥 Coordinator healthy: {health['coordinator_healthy']}")
    print(f"   🏥 Total agents: {health['total_agents']}")
    print(f"   🏥 Healthy agents: {health['healthy_agents']}")
    
    # 9. Cleanup
    print("\n9️⃣ Cleanup...")
    await registry.stop_all_agents()
    print("   ✅ All agents stopped")
    
    print("\n🎉 Demo completed successfully!")
    print("\n📋 Framework Features Demonstrated:")
    print("   • Agent registration and lifecycle management")
    print("   • Pipeline orchestration and execution")
    print("   • Performance monitoring and metrics")
    print("   • Health checking and error handling")
    print("   • Inter-agent communication and coordination")


if __name__ == "__main__":
    asyncio.run(demo_agent_framework()) 
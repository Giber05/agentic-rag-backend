"""
Enhanced API documentation configuration for FastAPI.

This module provides comprehensive API documentation setup including:
- Detailed OpenAPI metadata
- Example requests and responses
- API versioning information
- Contact and license information
"""

from typing import Dict, Any

# API Documentation Metadata
API_METADATA = {
    "title": "Agentic RAG AI Agent API",
    "description": """
## 🤖 Agentic RAG AI Agent Backend API

A sophisticated Retrieval-Augmented Generation (RAG) system featuring 5 specialized AI agents working in coordination to provide contextual, accurate responses with source attribution.

### 🏗️ Architecture Overview

The system consists of 5 main agents:

1. **Query Rewriting Agent** - Optimizes user queries for better retrieval
2. **Context Decision Agent** - Determines if additional context is needed  
3. **Source Retrieval Agent** - Retrieves relevant information from vector database
4. **Answer Generation Agent** - Generates responses using LLM with retrieved context
5. **RAG Pipeline Orchestrator** - Coordinates all agents in the pipeline

### 🚀 Key Features

- **Multi-Agent Coordination**: Sophisticated pipeline orchestration
- **Vector Search**: Semantic search using Supabase pgvector
- **Real-time Streaming**: WebSocket support for live responses
- **Source Attribution**: Comprehensive citation and source tracking
- **Performance Monitoring**: Detailed metrics and analytics
- **Caching System**: Intelligent caching for optimal performance
- **Error Handling**: Robust fallback strategies and error recovery

### 🔧 Technology Stack

- **Backend**: FastAPI (Python 3.11+)
- **Vector Database**: Supabase with pgvector extension
- **AI Integration**: OpenAI GPT-4-turbo and Embedding APIs
- **Caching**: Redis/In-memory caching
- **Real-time**: WebSocket connections

### 📚 API Usage

All endpoints are versioned under `/api/v1/` and return JSON responses. 
Authentication is handled via JWT tokens where required.

### 🔗 Quick Start

1. **Health Check**: `GET /health`
2. **Process Query**: `POST /api/v1/rag/process`
3. **Stream Response**: `WS /api/v1/rag/stream`
4. **Upload Document**: `POST /api/v1/documents/upload`

### 📊 Response Format

All API responses follow a consistent structure:

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "request_id": "uuid-string",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Error responses include detailed error information:

```json
{
  "error": "Error Type",
  "message": "Detailed error message",
  "request_id": "uuid-string",
  "timestamp": "2024-01-01T00:00:00Z"
}
```
    """,
    "version": "1.0.0",
    "contact": {
        "name": "Agentic RAG AI Agent Team",
        "email": "support@agentic-rag.com",
        "url": "https://github.com/your-org/agentic-rag-ai-agent"
    },
    "license_info": {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    "servers": [
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.agentic-rag.com",
            "description": "Production server"
        }
    ]
}

# Common response examples
COMMON_EXAMPLES = {
    "success_response": {
        "summary": "Successful Response",
        "value": {
            "success": True,
            "data": {"result": "Operation completed"},
            "message": "Request processed successfully",
            "request_id": "123e4567-e89b-12d3-a456-426614174000",
            "timestamp": "2024-01-01T12:00:00Z"
        }
    },
    "error_response": {
        "summary": "Error Response",
        "value": {
            "error": "ValidationError",
            "message": "Invalid input parameters",
            "request_id": "123e4567-e89b-12d3-a456-426614174000",
            "timestamp": "2024-01-01T12:00:00Z"
        }
    }
}

# Agent-specific examples
AGENT_EXAMPLES = {
    "query_rewriter": {
        "request": {
            "summary": "Query Rewriting Request",
            "value": {
                "query": "What is machine learning?",
                "options": {
                    "enable_spell_check": True,
                    "enable_grammar_check": True,
                    "enable_expansion": True
                }
            }
        },
        "response": {
            "summary": "Query Rewriting Response",
            "value": {
                "success": True,
                "data": {
                    "original_query": "What is machine learning?",
                    "rewritten_query": "What is machine learning and how does it work?",
                    "improvements": [
                        "Added clarifying context",
                        "Expanded for better retrieval"
                    ],
                    "confidence": 0.95,
                    "processing_time": 0.234
                },
                "message": "Query rewritten successfully",
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    },
    "context_decision": {
        "request": {
            "summary": "Context Decision Request",
            "value": {
                "query": "What is the capital of France?",
                "conversation_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help you?"}
                ]
            }
        },
        "response": {
            "summary": "Context Decision Response",
            "value": {
                "success": True,
                "data": {
                    "needs_context": False,
                    "confidence": 0.98,
                    "reasoning": "Query is factual and self-contained",
                    "decision_factors": {
                        "query_complexity": 0.2,
                        "context_dependency": 0.1,
                        "semantic_similarity": 0.05
                    }
                },
                "message": "Context decision completed",
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    },
    "source_retrieval": {
        "request": {
            "summary": "Source Retrieval Request",
            "value": {
                "query": "machine learning algorithms",
                "max_results": 5,
                "strategy": "semantic",
                "filters": {
                    "document_type": "pdf",
                    "date_range": "2023-2024"
                }
            }
        },
        "response": {
            "summary": "Source Retrieval Response",
            "value": {
                "success": True,
                "data": {
                    "sources": [
                        {
                            "id": "doc_123",
                            "title": "Introduction to Machine Learning",
                            "content": "Machine learning is a subset of artificial intelligence...",
                            "relevance_score": 0.95,
                            "metadata": {
                                "author": "John Doe",
                                "publication_date": "2023-06-15",
                                "document_type": "pdf"
                            }
                        }
                    ],
                    "total_found": 15,
                    "strategy_used": "semantic",
                    "processing_time": 0.456
                },
                "message": "Sources retrieved successfully",
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    },
    "answer_generation": {
        "request": {
            "summary": "Answer Generation Request",
            "value": {
                "query": "What is machine learning?",
                "sources": [
                    {
                        "title": "ML Basics",
                        "content": "Machine learning is a method of data analysis...",
                        "url": "https://example.com/ml-basics"
                    }
                ],
                "options": {
                    "citation_style": "numbered",
                    "response_format": "markdown",
                    "max_length": 500
                }
            }
        },
        "response": {
            "summary": "Answer Generation Response",
            "value": {
                "success": True,
                "data": {
                    "answer": "Machine learning is a method of data analysis that automates analytical model building [1]. It uses algorithms that iteratively learn from data...",
                    "citations": [
                        {
                            "id": 1,
                            "title": "ML Basics",
                            "url": "https://example.com/ml-basics"
                        }
                    ],
                    "quality_score": 0.92,
                    "confidence": 0.88,
                    "processing_time": 1.234
                },
                "message": "Answer generated successfully",
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    },
    "rag_pipeline": {
        "request": {
            "summary": "RAG Pipeline Request",
            "value": {
                "query": "Explain neural networks",
                "conversation_history": [],
                "options": {
                    "enable_streaming": False,
                    "citation_style": "numbered",
                    "max_sources": 5
                }
            }
        },
        "response": {
            "summary": "RAG Pipeline Response",
            "value": {
                "success": True,
                "data": {
                    "answer": "Neural networks are computing systems inspired by biological neural networks [1][2]. They consist of interconnected nodes (neurons) that process information...",
                    "sources": [
                        {
                            "id": "doc_456",
                            "title": "Neural Network Fundamentals",
                            "relevance_score": 0.94
                        }
                    ],
                    "pipeline_stages": {
                        "query_rewriting": {"status": "completed", "time": 0.234},
                        "context_decision": {"status": "completed", "time": 0.123},
                        "source_retrieval": {"status": "completed", "time": 0.456},
                        "answer_generation": {"status": "completed", "time": 1.234}
                    },
                    "total_processing_time": 2.047,
                    "quality_metrics": {
                        "relevance": 0.92,
                        "completeness": 0.88,
                        "accuracy": 0.95
                    }
                },
                "message": "RAG pipeline completed successfully",
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    }
}

# OpenAPI tags for better organization
OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Health check and system status endpoints"
    },
    {
        "name": "openai",
        "description": "OpenAI service integration and management"
    },
    {
        "name": "documents",
        "description": "Document upload, processing, and management"
    },
    {
        "name": "search",
        "description": "Vector search and retrieval operations"
    },
    {
        "name": "agents",
        "description": "Agent framework management and coordination"
    },
    {
        "name": "query-rewriter",
        "description": "Query optimization and rewriting agent"
    },
    {
        "name": "context-decision",
        "description": "Context necessity evaluation agent"
    },
    {
        "name": "source-retrieval",
        "description": "Source retrieval and ranking agent"
    },
    {
        "name": "answer-generation",
        "description": "Answer generation with citation agent"
    },
    {
        "name": "rag-pipeline",
        "description": "Complete RAG pipeline orchestration"
    }
]

def get_openapi_schema_extra() -> Dict[str, Any]:
    """Get additional OpenAPI schema configuration."""
    return {
        "info": {
            "x-logo": {
                "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
            }
        },
        "externalDocs": {
            "description": "GitHub Repository",
            "url": "https://github.com/your-org/agentic-rag-ai-agent"
        }
    } 
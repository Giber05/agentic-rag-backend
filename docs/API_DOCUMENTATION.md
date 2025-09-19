# 🤖 Agentic RAG AI Agent - API Documentation

## Overview

The Agentic RAG AI Agent backend provides a sophisticated Retrieval-Augmented Generation (RAG) system featuring 5 specialized AI agents working in coordination to provide contextual, accurate responses with source attribution. The system includes advanced cost optimization with 94% cost reduction, comprehensive token tracking, and multi-source retrieval capabilities including Jira integration via Model Context Protocol (MCP).

## 🚀 Quick Start

### 1. Access Documentation

- **Swagger UI**: `http://localhost:8000/api/v1/docs`
- **ReDoc**: `http://localhost:8000/api/v1/redoc`
- **OpenAPI JSON**: `http://localhost:8000/api/v1/openapi.json`

### 2. Import Postman Collection

Import the comprehensive Postman collection from `backend/docs/postman_collection.json` for interactive API testing.

### 3. Basic Usage

```bash
# Health check
curl http://localhost:8000/health

# Process query through optimized RAG pipeline (94% cost reduction)
curl -X POST http://localhost:8000/api/v1/rag/process \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "source": "db",
    "pipeline_config": {
      "citation_style": "numbered",
      "max_sources": 5,
      "enable_streaming": false
    }
  }'

# Process query with Jira sources
curl -X POST http://localhost:8000/api/v1/rag/process \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me recent authentication issues from TOCO project",
    "source": "jira",
    "pipeline_config": {
      "citation_style": "numbered",
      "max_sources": 10
    }
  }'

# Check token usage
curl http://localhost:8000/api/v1/analytics/recent-requests?limit=5
```

## 🏗️ Architecture Overview

### Agent Pipeline

1. **Query Rewriting Agent** - Optimizes user queries for better retrieval (✅ Optimized)
2. **Context Decision Agent** - Determines if additional context is needed (✅ Optimized)
3. **Source Retrieval Agent** - Retrieves relevant information from vector database (✅ Optimized)
4. **Answer Generation Agent** - Generates responses using LLM with retrieved context (✅ Optimized)
5. **Validation & Refinement Agent** - Validates and iteratively improves responses (❌ Not Implemented)

### Cost Optimization Features

- **Smart Agent Bypassing**: Skip unnecessary processing for simple queries
- **Model Downgrading**: Use GPT-3.5-turbo instead of GPT-4-turbo (20x cheaper)
- **Aggressive Caching**: 24-hour cache for repeated queries
- **Pattern Matching**: Direct answers for common greetings/thanks
- **Limited Source Retrieval**: Reduce sources from 10 to 5 for faster processing

### Technology Stack

- **Backend**: FastAPI (Python 3.13+)
- **Vector Database**: Supabase with pgvector extension
- **AI Integration**: OpenAI GPT-3.5-turbo, GPT-4-turbo, and Embedding APIs
- **Caching**: Redis/In-memory caching
- **Real-time**: WebSocket connections
- **Token Tracking**: Comprehensive analytics with cost monitoring

## 📚 API Reference

### Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.agentic-rag.com`

### API Version

All endpoints are versioned under `/api/v1/`

### Authentication

```http
Authorization: Bearer <jwt_token>
```

### Response Format

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

Error responses:

```json
{
  "error": "Error Type",
  "message": "Detailed error message",
  "request_id": "uuid-string",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 🔗 Endpoint Categories

### 🏥 Health & Status

#### `GET /health`

Check the health status of the API server.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600
}
```

#### `GET /api/v1/database/status`

Check the database connection status.

### 🤖 OpenAI Integration

#### `GET /api/v1/openai/health`

Check OpenAI service connectivity.

#### `GET /api/v1/openai/models`

Get list of available OpenAI models.

#### `POST /api/v1/openai/chat/completions`

Generate chat completion using OpenAI.

**Request:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is machine learning?"
    }
  ],
  "model": "gpt-3.5-turbo",
  "max_tokens": 150,
  "temperature": 0.7
}
```

#### `POST /api/v1/openai/embeddings`

Generate embeddings for text.

**Request:**

```json
{
  "input": "Machine learning is a subset of artificial intelligence",
  "model": "text-embedding-ada-002"
}
```

#### `GET /api/v1/openai/usage/stats`

Get OpenAI API usage statistics.

### 📊 Analytics & Token Tracking

#### `GET /api/v1/analytics/recent-requests`

Get recent request analytics with token usage.

**Query Parameters:**

- `limit`: Number of requests to return (default: 10, max: 100)

**Response:**

```json
{
  "success": true,
  "data": {
    "requests": [
      {
        "request_id": "123e4567-e89b-12d3-a456-426614174000",
        "timestamp": "2024-01-01T12:00:00Z",
        "query": "What is machine learning?",
        "total_tokens": 847,
        "total_cost": 0.0007,
        "pipeline_type": "optimized",
        "processing_time": 2.34
      }
    ],
    "total_requests": 25,
    "total_tokens": 21250,
    "total_cost": 0.0175
  }
}
```

#### `GET /api/v1/analytics/token-usage/{request_id}`

Get detailed token breakdown for a specific request.

**Response:**

```json
{
  "success": true,
  "data": {
    "request_id": "123e4567-e89b-12d3-a456-426614174000",
    "query": "What is machine learning?",
    "total_tokens": 847,
    "total_cost": 0.0007,
    "breakdown": {
      "query_rewriting": {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "skipped": true
      },
      "embedding": {
        "input_tokens": 25,
        "output_tokens": 0,
        "cost": 0.0000025,
        "model": "text-embedding-ada-002"
      },
      "answer_generation": {
        "input_tokens": 645,
        "output_tokens": 177,
        "cost": 0.0006,
        "model": "gpt-3.5-turbo"
      }
    },
    "optimization_savings": {
      "estimated_full_cost": 0.0588,
      "actual_cost": 0.0007,
      "savings_percentage": 94.5
    }
  }
}
```

#### `GET /api/v1/analytics/daily-stats`

Get daily usage statistics.

**Response:**

```json
{
  "success": true,
  "data": {
    "date": "2024-01-01",
    "total_requests": 152,
    "total_tokens": 128450,
    "total_cost": 0.89,
    "average_tokens_per_request": 845,
    "average_cost_per_request": 0.0058,
    "optimization_stats": {
      "optimized_requests": 140,
      "full_pipeline_requests": 12,
      "total_savings": 0.78,
      "savings_percentage": 87.6
    }
  }
}
```

#### `GET /api/v1/analytics/cost-patterns`

Get cost analysis and patterns.

**Response:**

```json
{
  "success": true,
  "data": {
    "cost_by_operation": {
      "query_rewriting": 0.05,
      "embedding": 0.12,
      "answer_generation": 0.72
    },
    "cost_by_model": {
      "gpt-3.5-turbo": 0.65,
      "gpt-4-turbo": 0.15,
      "text-embedding-ada-002": 0.09
    },
    "optimization_impact": {
      "pattern_matching_savings": 0.08,
      "agent_bypassing_savings": 0.04,
      "model_downgrading_savings": 0.055
    }
  }
}
```

#### `GET /api/v1/analytics/monthly-projection`

Get monthly cost projection based on current usage.

**Response:**

```json
{
  "success": true,
  "data": {
    "current_month_cost": 15.67,
    "projected_month_cost": 18.45,
    "without_optimization_projection": 245.8,
    "savings_projection": 227.35,
    "daily_average": 0.89,
    "requests_per_day": 152
  }
}
```

### 📄 Document Management

#### `POST /api/v1/documents/upload`

Upload and process a document.

**Request:** `multipart/form-data`

- `file`: Document file
- `title`: Document title
- `description`: Document description

#### `GET /api/v1/documents`

Get list of uploaded documents.

**Query Parameters:**

- `limit`: Number of documents to return (default: 10)
- `offset`: Number of documents to skip (default: 0)

#### `GET /api/v1/documents/{id}`

Get details of a specific document.

#### `DELETE /api/v1/documents/{id}`

Delete a document and its embeddings.

### 🔍 Search & Retrieval

#### `POST /api/v1/search/semantic`

Perform semantic search using vector similarity.

**Request:**

```json
{
  "query": "machine learning algorithms",
  "limit": 5,
  "threshold": 0.7
}
```

#### `POST /api/v1/search/hybrid`

Perform hybrid search combining semantic and keyword search.

**Request:**

```json
{
  "query": "neural networks deep learning",
  "limit": 10,
  "semantic_weight": 0.7,
  "keyword_weight": 0.3
}
```

### 🎯 Agent Framework

#### `GET /api/v1/agents`

Get list of all registered agents.

#### `GET /api/v1/agents/{id}`

Get details of a specific agent.

#### `GET /api/v1/agents/metrics`

Get performance metrics for all agents.

### 🔗 MCP Integration (Jira)

#### `GET /api/v1/mcp/jira/health`

Check Jira MCP service connectivity and configuration.

**Response:**

```json
{
  "success": true,
  "data": {
    "status": "connected",
    "jira_url": "https://sprout-id.atlassian.net",
    "connection_verified": true,
    "projects_count": 6,
    "last_check": "2024-01-01T12:00:00Z"
  }
}
```

#### `GET /api/v1/mcp/jira/projects`

Get list of available Jira projects.

**Response:**

```json
{
  "success": true,
  "data": {
    "projects": [
      {
        "key": "TOCO",
        "name": "TOCO",
        "id": "10000"
      },
      {
        "key": "SPEC",
        "name": "Spectra",
        "id": "10001"
      }
    ],
    "total_count": 6
  }
}
```

#### `GET /api/v1/mcp/jira/search`

Search Jira issues using natural language queries.

**Query Parameters:**

- `query`: Natural language search query (required)
- `max_results`: Maximum number of results to return (default: 10, max: 50)

**Example Requests:**

```bash
# Search for recent issues from TOCO project
GET /api/v1/mcp/jira/search?query=recent%20issues%20from%20TOCO%20project&max_results=5

# Search for Confluence documentation
GET /api/v1/mcp/jira/search?query=confluence%20wiki%20documentation&max_results=10

# Search for specific issue
GET /api/v1/mcp/jira/search?query=TOCO-7445&max_results=1
```

**Response:**

```json
{
  "success": true,
  "data": {
    "issues": [
      {
        "key": "TOCO-7445",
        "summary": "Fix authentication bug in mobile app",
        "description": "Users are experiencing login issues...",
        "status": "In Progress",
        "priority": "High",
        "assignee": "john.doe@company.com",
        "created": "2024-01-01T10:00:00Z",
        "updated": "2024-01-01T14:30:00Z",
        "url": "https://sprout-id.atlassian.net/browse/TOCO-7445",
        "project": {
          "key": "TOCO",
          "name": "TOCO"
        }
      }
    ],
    "total_found": 127,
    "jql_used": "project = \"TOCO\" AND text ~ \"recent\" ORDER BY created DESC",
    "query_processed": "recent issues from TOCO project"
  }
}
```

#### `POST /api/v1/mcp/jira/search`

Advanced Jira search with custom JQL query.

**Request:**

```json
{
  "jql": "project = \"TOCO\" AND status = \"In Progress\" ORDER BY priority DESC",
  "limit": 10
}
```

### ✏️ Query Rewriter Agent

#### `POST /api/v1/query-rewriter/process`

Process and rewrite a query for better retrieval.

**Request:**

```json
{
  "query": "What is machine learning?",
  "options": {
    "enable_spell_check": true,
    "enable_grammar_check": true,
    "enable_expansion": true
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "original_query": "What is machine learning?",
    "rewritten_query": "What is machine learning and how does it work?",
    "improvements": [
      "Added clarifying context",
      "Expanded for better retrieval"
    ],
    "confidence": 0.95,
    "processing_time": 0.234,
    "skipped": false,
    "skip_reason": null
  }
}
```

#### `POST /api/v1/query-rewriter/agent/create`

Create a new Query Rewriter agent instance.

#### `GET /api/v1/query-rewriter/stats`

Get performance statistics for Query Rewriter agent.

### 🤔 Context Decision Agent

#### `POST /api/v1/context-decision/evaluate`

Evaluate whether additional context is needed for a query.

**Request:**

```json
{
  "query": "What is the capital of France?",
  "conversation_history": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help you?"
    }
  ]
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "needs_context": false,
    "confidence": 0.98,
    "reasoning": "Query is factual and self-contained",
    "method": "rule_based",
    "decision_factors": {
      "query_complexity": 0.2,
      "context_dependency": 0.1,
      "semantic_similarity": 0.05
    },
    "processing_time": 0.001
  }
}
```

#### `GET /api/v1/context-decision/metrics`

Get decision accuracy and performance metrics.

#### `POST /api/v1/context-decision/agent/create`

Create a new Context Decision agent instance.

### 📚 Enhanced Source Retrieval Agent

#### `POST /api/v1/source-retrieval/retrieve`

Retrieve relevant sources for a query from multiple sources including vector database and Jira via MCP integration.

**Request:**

```json
{
  "query": "machine learning algorithms",
  "source": "db",
  "max_results": 5,
  "strategy": "semantic",
  "filters": {
    "document_type": "pdf",
    "date_range": "2023-2024"
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "sources": [
      {
        "id": "doc_123",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence...",
        "relevance_score": 0.95,
        "source_type": "document",
        "metadata": {
          "author": "John Doe",
          "publication_date": "2023-06-15",
          "document_type": "pdf"
        }
      },
      {
        "id": "TOCO-7445",
        "title": "TOCO-7445: Implement ML algorithms for user behavior analysis",
        "content": "We need to implement machine learning algorithms to analyze user behavior patterns...",
        "relevance_score": 0.89,
        "source_type": "jira",
        "metadata": {
          "project": "TOCO",
          "status": "In Progress",
          "assignee": "jane.smith@company.com",
          "created": "2024-01-01T10:00:00Z",
          "url": "https://sprout-id.atlassian.net/browse/TOCO-7445"
        }
      }
    ],
    "total_found": 15,
    "strategy_used": "hybrid_limited",
    "processing_time": 0.456,
    "cache_hit": false
  }
}
```

#### `GET /api/v1/source-retrieval/performance`

Get retrieval performance metrics.

#### `GET /api/v1/source-retrieval/strategies`

Get list of available retrieval strategies.

### 💬 Answer Generation Agent

#### `POST /api/v1/answer-generation/generate`

Generate an answer with source citations.

**Request:**

```json
{
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
    "max_length": 500,
    "model": "gpt-3.5-turbo"
  }
}
```

**Response:**

```json
{
  "success": true,
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
    "processing_time": 1.234,
    "model_used": "gpt-3.5-turbo",
    "token_usage": {
      "input_tokens": 645,
      "output_tokens": 177,
      "total_cost": 0.0006
    }
  }
}
```

#### `POST /api/v1/answer-generation/stream`

Stream answer generation in real-time.

#### `GET /api/v1/answer-generation/performance`

Get answer quality and performance metrics.

### 🔄 RAG Pipeline

#### `POST /api/v1/rag/process` (Optimized Pipeline - Default)

Process a query through the optimized RAG pipeline with 94% cost reduction and multi-source support.

**Request:**

```json
{
  "query": "Explain neural networks",
  "source": "db",
  "conversation_history": [],
  "pipeline_config": {
    "enable_streaming": false,
    "citation_style": "numbered",
    "max_sources": 5,
    "response_format": "markdown"
  }
}
```

**Source Options:**

- `"db"` (default): Search vector database only
- `"jira"`: Search Jira issues only via MCP integration
- `"jira&db"`: Search both Jira and database sources (combined)

**Example Jira Query:**

```json
{
  "query": "Show me recent issues from TOCO project about authentication bugs",
  "source": "jira",
  "pipeline_config": {
    "max_sources": 10,
    "citation_style": "numbered"
  }
}
```

**Example Combined Sources:**

```json
{
  "query": "What are the best practices for authentication implementation?",
  "source": "jira&db",
  "pipeline_config": {
    "max_sources": 15,
    "citation_style": "numbered"
  }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "request_id": "123e4567-e89b-12d3-a456-426614174000",
    "query": "Explain neural networks",
    "status": "completed",
    "pipeline_type": "optimized",
    "final_response": {
      "query": "Explain neural networks",
      "response": {
        "content": "Neural networks are computing systems inspired by biological neural networks [1][2]. They consist of interconnected nodes (neurons) that process information...",
        "citations": [
          {
            "id": 1,
            "source_id": "doc_456",
            "title": "Neural Network Fundamentals",
            "relevance_score": 0.94,
            "source_type": "document",
            "url": "https://example.com/neural-networks"
          },
          {
            "id": 2,
            "source_id": "TOCO-7445",
            "title": "TOCO-7445: Implement neural network authentication",
            "relevance_score": 0.87,
            "source_type": "jira",
            "url": "https://sprout-id.atlassian.net/browse/TOCO-7445"
          }
        ],
        "quality": {
          "relevance_score": 0.92,
          "coherence_score": 0.88,
          "completeness_score": 0.95,
          "overall_quality": 0.92
        },
        "metadata": {
          "model_used": "gpt-3.5-turbo",
          "sources_count": 3,
          "word_count": 245
        }
      },
      "sources_used": 3
    },
    "stage_results": {
      "query_rewriting": {
        "status": "skipped",
        "reason": "simple_query",
        "duration": 0.001
      },
      "context_decision": {
        "status": "completed",
        "method": "rule_based",
        "duration": 0.002
      },
      "source_retrieval": {
        "status": "completed",
        "strategy": "hybrid_limited",
        "duration": 0.456
      },
      "answer_generation": {
        "status": "completed",
        "model_used": "gpt-3.5-turbo",
        "duration": 1.234
      }
    },
    "total_duration": 1.693,
    "optimization_info": {
      "pipeline_used": "optimized",
      "cost_savings": 0.0581,
      "savings_percentage": 94.5
    },
    "token_usage": {
      "total_tokens": 847,
      "total_cost": 0.0007,
      "breakdown": {
        "embedding": { "tokens": 25, "cost": 0.0000025 },
        "answer_generation": { "tokens": 822, "cost": 0.0007 }
      }
    }
  }
}
```

#### `POST /api/v1/rag/process/full` (Full Pipeline)

Process a query through the complete RAG pipeline with all agents.

**Request:**

```json
{
  "query": "Complex query requiring full processing",
  "conversation_history": [],
  "pipeline_config": {
    "enable_streaming": false,
    "citation_style": "numbered",
    "max_sources": 10,
    "use_premium_models": true
  }
}
```

#### `POST /api/v1/rag/stream`

Stream RAG pipeline response in real-time with multi-source support.

**Request:**

```json
{
  "query": "Show me recent Jira issues about authentication",
  "source": "jira",
  "pipeline_config": {
    "enable_streaming": true,
    "max_sources": 10
  }
}
```

#### `GET /api/v1/rag/pipeline/status`

Get current pipeline status and health.

**Response:**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "active_agents": 4,
    "optimization_enabled": true,
    "cache_status": "active",
    "recent_performance": {
      "avg_response_time": 1.85,
      "success_rate": 99.2,
      "optimization_rate": 87.5
    }
  }
}
```

#### `GET /api/v1/rag/pipeline/metrics`

Get comprehensive pipeline performance metrics.

**Response:**

```json
{
  "success": true,
  "data": {
    "total_requests": 1250,
    "optimized_requests": 1094,
    "full_pipeline_requests": 156,
    "average_response_time": 1.85,
    "cost_metrics": {
      "total_cost": 12.45,
      "average_cost_per_request": 0.0099,
      "total_savings": 156.78,
      "savings_percentage": 92.6
    },
    "agent_performance": {
      "query_rewriter": { "bypass_rate": 0.75, "avg_time": 0.234 },
      "context_decision": { "accuracy": 0.94, "avg_time": 0.002 },
      "source_retrieval": { "cache_hit_rate": 0.35, "avg_time": 0.456 },
      "answer_generation": { "quality_score": 0.89, "avg_time": 1.234 }
    }
  }
}
```

## 🔌 WebSocket Endpoints

### RAG Pipeline Streaming

Connect to `ws://localhost:8000/api/v1/rag/stream` for real-time pipeline updates.

**Message Format:**

```json
{
  "type": "stage_update",
  "stage": "query_rewriting",
  "status": "processing",
  "data": { ... },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 📊 Response Codes

| Code | Description           |
| ---- | --------------------- |
| 200  | Success               |
| 201  | Created               |
| 400  | Bad Request           |
| 401  | Unauthorized          |
| 403  | Forbidden             |
| 404  | Not Found             |
| 422  | Validation Error      |
| 429  | Rate Limited          |
| 500  | Internal Server Error |

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Optimization Settings
ENABLE_COST_OPTIMIZATION=true
ENABLE_AGGRESSIVE_CACHING=true
DEFAULT_PIPELINE=optimized
```

### Pipeline Configuration

```json
{
  "optimized_pipeline": {
    "enable_smart_bypassing": true,
    "use_cheap_models": true,
    "max_sources": 5,
    "cache_ttl": 86400,
    "default_model": "gpt-3.5-turbo"
  },
  "full_pipeline": {
    "enable_all_agents": true,
    "use_premium_models": true,
    "max_sources": 10,
    "default_model": "gpt-4-turbo"
  }
}
```

## 🧪 Testing

### Using Postman

1. Import the collection from `backend/docs/postman_collection.json`
2. Set environment variables:
   - `base_url`: `http://localhost:8000`
   - `jwt_token`: Your authentication token (if required)
3. Run the collection or individual requests

### Using curl

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test optimized RAG pipeline
curl -X POST http://localhost:8000/api/v1/rag/process \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "pipeline_config": {
      "citation_style": "numbered",
      "max_sources": 5
    }
  }'

# Check token usage
curl http://localhost:8000/api/v1/analytics/recent-requests?limit=5
```

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Optimized RAG pipeline
response = requests.post(
    "http://localhost:8000/api/v1/rag/process",
    json={
        "query": "What is machine learning?",
        "pipeline_config": {
            "citation_style": "numbered",
            "max_sources": 5
        }
    }
)
print(response.json())

# Check analytics
response = requests.get("http://localhost:8000/api/v1/analytics/daily-stats")
print(response.json())
```

## 🚀 Performance

### Expected Response Times

| Endpoint              | Optimized | Full Pipeline |
| --------------------- | --------- | ------------- |
| Health Check          | < 50ms    | < 50ms        |
| Query Rewriting       | < 10ms    | < 500ms       |
| Context Decision      | < 5ms     | < 200ms       |
| Source Retrieval      | < 500ms   | < 800ms       |
| Answer Generation     | < 1500ms  | < 2000ms      |
| Complete RAG Pipeline | < 2000ms  | < 3000ms      |

### Cost Comparison

| Pipeline Type | Cost per Request | Tokens per Request | Savings |
| ------------- | ---------------- | ------------------ | ------- |
| Optimized     | ~$0.0007         | ~800               | 94%     |
| Full          | ~$0.10+          | ~2600+             | 0%      |

### Rate Limits

- **General API**: 100 requests/minute
- **RAG Pipeline**: 10 requests/minute
- **Document Upload**: 5 requests/minute
- **Analytics**: 50 requests/minute

## 🔒 Security

### Authentication

JWT tokens are required for protected endpoints. Include in the Authorization header:

```http
Authorization: Bearer <your_jwt_token>
```

### Input Validation

All inputs are validated using Pydantic models. Invalid requests return 422 status with detailed error messages.

### Rate Limiting

API endpoints are protected with rate limiting to prevent abuse.

## 🐛 Error Handling

### Common Error Responses

```json
{
  "error": "ValidationError",
  "message": "Invalid input parameters",
  "details": {
    "field": "query",
    "issue": "Field required"
  },
  "request_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

### Troubleshooting

1. **500 Internal Server Error**: Check server logs and OpenAI API key
2. **422 Validation Error**: Verify request body format and required fields
3. **404 Not Found**: Ensure correct endpoint URL and API version
4. **429 Rate Limited**: Wait before making additional requests
5. **High Token Usage**: Check if using optimized pipeline (`/api/v1/rag/process`)

## 📞 Support

- **Documentation**: Check Swagger UI at `/api/v1/docs`
- **Issues**: Report bugs on GitHub
- **Email**: support@agentic-rag.com

## 📝 Changelog

### v1.1.0 (Current)

- ✅ **Cost Optimization**: 94% cost reduction with optimized pipeline
- ✅ **Token Tracking**: Comprehensive analytics and monitoring
- ✅ **Smart Agent Bypassing**: Skip unnecessary processing
- ✅ **Model Optimization**: Use GPT-3.5-turbo for most tasks
- ✅ **Aggressive Caching**: 24-hour cache for repeated queries
- ✅ **Analytics Endpoints**: Detailed usage and cost analysis

### v1.0.0

- Initial release with complete RAG pipeline
- 5 specialized agents implementation
- WebSocket streaming support
- Comprehensive API documentation
- Postman collection included

---

_For more detailed information, visit the interactive Swagger documentation at `/api/v1/docs` when the server is running._

# RAG Process Sequence Diagrams

## 🎯 Overview

This document provides detailed sequence diagrams showing the complete RAG process flow from API request to final response. It covers both optimized and full pipeline variants, including all service interactions and data flows.

## 🔄 Complete RAG Process Sequence Diagram

The following diagram shows the complete flow of a RAG request through the optimized pipeline:

```mermaid
sequenceDiagram
    participant Client
    participant API as "FastAPI Endpoint"
    participant OptOrch as "OptimizedRAGOrchestrator"
    participant Cache as "Cache Service"
    participant TokenTracker as "Token Tracker"
    participant ContextAgent as "Context Decision Agent"
    participant VectorSearch as "Vector Search Service"
    participant Supabase as "Supabase Vector DB"
    participant OpenAI as "OpenAI API"
    participant AnswerAgent as "Answer Generation Agent"

    Client->>API: POST /api/v1/rag/process
    Note over Client,API: RAGRequest with query, history, context

    API->>OptOrch: Dependency Injection
    API->>OptOrch: process_query(request)

    OptOrch->>TokenTracker: start_request_tracking()
    OptOrch->>Cache: check cache for query

    alt Cache Hit
        Cache-->>OptOrch: cached result
        OptOrch-->>API: ProcessingResult (cached)
        API-->>Client: Response (saved ~$0.08)
    else Cache Miss
        OptOrch->>OptOrch: pattern matching check

        alt Non-informational pattern (greeting/thanks)
            OptOrch-->>API: ProcessingResult (pattern response)
            API-->>Client: Response (saved ~$0.06)
        else Informational query
            OptOrch->>OptOrch: determine model complexity

            Note over OptOrch: Stage 1: Context Decision
            OptOrch->>ContextAgent: process(query, history)
            ContextAgent->>OpenAI: analyze context necessity
            OpenAI-->>ContextAgent: decision + confidence
            ContextAgent-->>OptOrch: context_needed=true/false

            alt Context needed
                Note over OptOrch: Stage 2: Source Retrieval
                OptOrch->>VectorSearch: semantic_search(query)
                VectorSearch->>OpenAI: create_embedding(query)
                OpenAI-->>VectorSearch: query_embedding
                VectorSearch->>Supabase: vector similarity search
                Supabase-->>VectorSearch: relevant documents
                VectorSearch-->>OptOrch: sources[]
            else Context not needed
                OptOrch->>OptOrch: skip retrieval, sources=[]
            end

            Note over OptOrch: Stage 3: Answer Generation
            OptOrch->>AnswerAgent: process(query, sources, history)
            AnswerAgent->>OpenAI: chat completion (GPT-3.5/GPT-4)
            OpenAI-->>AnswerAgent: generated response
            AnswerAgent-->>OptOrch: final_response with citations

            OptOrch->>TokenTracker: track_api_calls()
            OptOrch->>Cache: store result
            OptOrch->>TokenTracker: finish_request_tracking()

            OptOrch-->>API: ProcessingResult
            API-->>Client: Response with answer + citations
        end
    end
```

## 🏗️ Dependency Injection Flow

This diagram shows how dependencies are injected and initialized:

```mermaid
sequenceDiagram
    participant FastAPI
    participant Dependencies as "dependencies.py"
    participant Supabase
    participant OpenAIService
    participant AgentRegistry
    participant OptOrch as "OptimizedRAGOrchestrator"

    FastAPI->>Dependencies: get_optimized_orchestrator()

    alt First call (not cached)
        Dependencies->>Dependencies: get_agent_registry()
        Dependencies->>AgentRegistry: create instance
        AgentRegistry->>AgentRegistry: register agent types

        Dependencies->>Dependencies: get_openai_service()
        Dependencies->>OpenAIService: create instance
        OpenAIService->>OpenAI: initialize client

        Dependencies->>Dependencies: get_supabase_client()
        Dependencies->>Supabase: create_client()

        Dependencies->>OptOrch: create instance
        OptOrch->>OptOrch: initialize config

        Dependencies-->>FastAPI: orchestrator instance
    else Subsequent calls (cached)
        Dependencies-->>FastAPI: cached orchestrator
    end
```

## 🔍 Vector Search Detail Flow

This diagram focuses on the vector search process within source retrieval:

```mermaid
sequenceDiagram
    participant OptOrch as "Orchestrator"
    participant VectorSearch as "VectorSearchService"
    participant OpenAI as "OpenAI API"
    participant Supabase as "Supabase Vector DB"
    participant Cache as "Cache Service"

    OptOrch->>VectorSearch: semantic_search(query, config)

    VectorSearch->>Cache: check embedding cache
    alt Embedding cached
        Cache-->>VectorSearch: cached embedding
    else Embedding not cached
        VectorSearch->>OpenAI: create_embedding(query)
        OpenAI-->>VectorSearch: query_embedding[1536]
        VectorSearch->>Cache: store embedding
    end

    VectorSearch->>Supabase: RPC search_embeddings()
    Note over VectorSearch,Supabase: Vector similarity search with cosine distance
    Supabase-->>VectorSearch: ranked results with scores

    VectorSearch->>VectorSearch: post_process_results()
    VectorSearch->>VectorSearch: apply_recency_boost()
    VectorSearch->>VectorSearch: filter by threshold

    VectorSearch-->>OptOrch: SearchResult[] with metadata
```

## 📊 Full Pipeline vs Optimized Pipeline

This diagram compares the two pipeline variants:

```mermaid
graph TD
    A[Client Request] --> B{Pipeline Type?}

    B -->|use_full_pipeline=true| C[RAGPipelineOrchestrator]
    B -->|use_full_pipeline=false| D[OptimizedRAGPipelineOrchestrator]

    C --> E[Full Query Rewriting]
    C --> F[Full Context Decision]
    C --> G[Full Source Retrieval]
    C --> H[Full Answer Generation]

    D --> I[Smart Caching Check]
    I --> J{Cache Hit?}
    J -->|Yes| K[Return Cached - $0.08 saved]
    J -->|No| L[Pattern Matching]
    L --> M{Non-informational?}
    M -->|Yes| N[Pattern Response - $0.06 saved]
    M -->|No| O[Smart Model Selection]
    O --> P[Optimized Context Decision]
    O --> Q[Quality Source Retrieval]
    O --> R[Smart Answer Generation]

    E --> S[Standard Response]
    F --> S
    G --> S
    H --> S

    K --> T[Client Response]
    N --> T
    P --> U[Optimized Response]
    Q --> U
    R --> U
    S --> T
    U --> T
```

## 🔄 Streaming Response Flow

This diagram shows the streaming endpoint behavior:

```mermaid
sequenceDiagram
    participant Client
    participant API as "Streaming Endpoint"
    participant Orchestrator
    participant Agents as "RAG Agents"

    Client->>API: POST /api/v1/rag/stream
    API->>Orchestrator: stream_query()

    Orchestrator->>Client: {"stage": "query_rewriting", "status": "starting"}
    Orchestrator->>Agents: Query Rewriting Agent
    Agents-->>Orchestrator: rewritten query
    Orchestrator->>Client: {"stage": "query_rewriting", "status": "completed"}

    Orchestrator->>Client: {"stage": "context_decision", "status": "processing"}
    Orchestrator->>Agents: Context Decision Agent
    Agents-->>Orchestrator: context decision
    Orchestrator->>Client: {"stage": "context_decision", "status": "completed"}

    alt Context needed
        Orchestrator->>Client: {"stage": "source_retrieval", "status": "processing"}
        Orchestrator->>Agents: Source Retrieval Agent
        Agents-->>Orchestrator: sources
        Orchestrator->>Client: {"stage": "source_retrieval", "status": "completed"}
    else Context not needed
        Orchestrator->>Client: {"stage": "source_retrieval", "status": "skipped"}
    end

    Orchestrator->>Client: {"stage": "answer_generation", "status": "processing"}
    Orchestrator->>Agents: Answer Generation Agent (streaming)

    loop Streaming chunks
        Agents-->>Orchestrator: response chunk
        Orchestrator->>Client: {"stage": "answer_generation", "chunk": "..."}
    end

    Agents-->>Orchestrator: final response
    Orchestrator->>Client: {"stage": "completed", "final_response": {...}}
```

## 🚨 Error Handling Flow

This diagram shows error handling and fallback mechanisms:

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Orchestrator
    participant Agent
    participant OpenAI
    participant Fallback as "Fallback Handler"

    Client->>API: RAG Request
    API->>Orchestrator: process_query()

    Orchestrator->>Agent: process stage
    Agent->>OpenAI: API call

    alt API Success
        OpenAI-->>Agent: successful response
        Agent-->>Orchestrator: stage result
        Orchestrator-->>API: ProcessingResult
        API-->>Client: Success Response
    else API Failure
        OpenAI-->>Agent: error (rate limit/timeout)
        Agent->>Agent: retry with backoff

        alt Retry Success
            OpenAI-->>Agent: successful response
            Agent-->>Orchestrator: stage result
        else Retry Failed
            Agent->>Fallback: handle_stage_error()
            Fallback-->>Agent: fallback response
            Agent-->>Orchestrator: fallback result
        end

        Orchestrator-->>API: ProcessingResult (with fallback)
        API-->>Client: Response with error info
    end
```

## 📈 Performance Monitoring Flow

This diagram shows how performance metrics are collected:

```mermaid
sequenceDiagram
    participant Request
    participant TokenTracker
    participant Metrics as "Performance Metrics"
    participant Cache
    participant Database as "Analytics DB"

    Request->>TokenTracker: start_request_tracking()
    TokenTracker->>Metrics: initialize request metrics

    loop Each API Call
        Request->>TokenTracker: track_api_call()
        TokenTracker->>Metrics: update token usage
        TokenTracker->>Metrics: calculate costs
    end

    Request->>TokenTracker: finish_request_tracking()
    TokenTracker->>Metrics: finalize request analysis

    TokenTracker->>Cache: cache performance data
    TokenTracker->>Database: store analytics

    Note over TokenTracker,Database: Cost optimization insights
    Note over TokenTracker,Database: Usage patterns analysis
```

## 🔧 Service Integration Overview

This diagram shows how all services integrate:

```mermaid
graph TB
    subgraph "API Layer"
        API[FastAPI Endpoints]
    end

    subgraph "Orchestration Layer"
        OptOrch[OptimizedRAGOrchestrator]
        FullOrch[RAGPipelineOrchestrator]
    end

    subgraph "Agent Layer"
        QR[Query Rewriter]
        CD[Context Decision]
        SR[Source Retrieval]
        AG[Answer Generation]
    end

    subgraph "Service Layer"
        OpenAIService[OpenAI Service]
        VectorSearch[Vector Search]
        DocumentService[Document Service]
        CacheService[Cache Service]
        TokenTracker[Token Tracker]
    end

    subgraph "Data Layer"
        Supabase[(Supabase Vector DB)]
        Redis[(Redis Cache)]
        OpenAI_API[OpenAI API]
    end

    API --> OptOrch
    API --> FullOrch

    OptOrch --> QR
    OptOrch --> CD
    OptOrch --> SR
    OptOrch --> AG

    FullOrch --> QR
    FullOrch --> CD
    FullOrch --> SR
    FullOrch --> AG

    QR --> OpenAIService
    CD --> OpenAIService
    SR --> VectorSearch
    AG --> OpenAIService

    VectorSearch --> DocumentService
    OpenAIService --> CacheService
    OptOrch --> TokenTracker

    OpenAIService --> OpenAI_API
    VectorSearch --> Supabase
    CacheService --> Redis
    DocumentService --> Supabase
```

## 📝 Key Insights from Sequence Diagrams

### Performance Optimizations

1. **Caching Strategy**: Multiple cache layers reduce API calls
2. **Pattern Matching**: Handles simple queries without AI processing
3. **Smart Model Selection**: Uses appropriate model for query complexity
4. **Dependency Injection**: Efficient resource management

### Cost Reduction Mechanisms

1. **Aggressive Caching**: 24-hour TTL saves ~$0.08 per cache hit
2. **Pattern Responses**: Non-informational queries save ~$0.06
3. **Model Optimization**: GPT-3.5 vs GPT-4 selection
4. **Token Tracking**: Real-time cost monitoring

### Error Resilience

1. **Retry Logic**: Exponential backoff for API failures
2. **Fallback Responses**: Graceful degradation
3. **Stage Isolation**: Failures don't cascade
4. **Monitoring**: Comprehensive error tracking

### Scalability Features

1. **Async Processing**: Non-blocking operations
2. **Connection Pooling**: Efficient database connections
3. **Rate Limiting**: API quota management
4. **Streaming Support**: Real-time responses

These sequence diagrams provide a comprehensive view of the RAG system's architecture and data flow, enabling developers to understand the complete request lifecycle and optimization strategies.

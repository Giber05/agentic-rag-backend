# RAG Services and Models Documentation

## 🎯 Overview

This document provides detailed information about all services and models involved in the RAG process, including their responsibilities, interfaces, and interactions.

## 🏗️ Core Services Architecture

### 1. OpenAI Service (`services/openai_service.py`)

**Purpose**: Centralized interface for all OpenAI API interactions with intelligent caching and error handling.

**Key Features**:

- **Smart Caching**: 24-hour TTL for embeddings and responses
- **Rate Limiting**: Automatic backoff and retry logic
- **Cost Tracking**: Real-time token usage monitoring
- **Model Selection**: Dynamic model switching based on complexity

**Core Methods**:

```python
class OpenAIService:
    async def create_embedding(self, text: str) -> List[float]
    async def chat_completion(self, messages: List[Dict], model: str = None) -> str
    async def chat_completion_stream(self, messages: List[Dict]) -> AsyncGenerator
    async def get_cached_response(self, cache_key: str) -> Optional[str]
    async def cache_response(self, cache_key: str, response: str, ttl: int = 86400)
```

**Integration Points**:

- Used by all RAG agents for AI processing
- Integrates with `CacheService` for response caching
- Monitored by `TokenTracker` for cost analysis

### 2. Vector Search Service (`services/vector_search_service.py`)

**Purpose**: Handles semantic search operations against the Supabase vector database.

**Key Features**:

- **Embedding Caching**: Reduces redundant embedding API calls
- **Similarity Search**: Cosine distance-based retrieval
- **Result Post-processing**: Recency boost and relevance filtering
- **Configurable Parameters**: Adjustable similarity thresholds

**Core Methods**:

```python
class VectorSearchService:
    async def semantic_search(
        self,
        query: str,
        config: SearchConfig
    ) -> List[SearchResult]

    async def create_and_cache_embedding(self, text: str) -> List[float]
    async def search_similar_embeddings(
        self,
        embedding: List[float],
        config: SearchConfig
    ) -> List[Dict]

    def post_process_results(
        self,
        results: List[Dict],
        config: SearchConfig
    ) -> List[SearchResult]
```

**Search Configuration**:

```python
@dataclass
class SearchConfig:
    similarity_threshold: float = 0.7
    max_results: int = 10
    recency_boost_factor: float = 0.1
    include_metadata: bool = True
```

### 3. Cache Service (`services/cache_service.py`)

**Purpose**: Provides intelligent caching for RAG responses and embeddings.

**Key Features**:

- **Multi-level Caching**: Response cache and embedding cache
- **TTL Management**: Configurable expiration times
- **Cache Invalidation**: Smart cache clearing strategies
- **Memory Optimization**: LRU eviction policies

**Core Methods**:

```python
class CacheService:
    async def get_cached_response(self, query_hash: str) -> Optional[ProcessingResult]
    async def cache_response(
        self,
        query_hash: str,
        result: ProcessingResult,
        ttl: int = 86400
    )

    async def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]
    async def cache_embedding(
        self,
        text_hash: str,
        embedding: List[float],
        ttl: int = 604800
    )

    def generate_query_hash(self, request: RAGRequest) -> str
    async def clear_cache(self, pattern: str = None)
```

**Cache Strategy**:

- **Response Cache**: 24-hour TTL, saves ~$0.08 per hit
- **Embedding Cache**: 7-day TTL, saves ~$0.0004 per embedding
- **Pattern Matching**: Immediate responses for greetings/thanks

### 4. Document Service (`services/document_service.py`)

**Purpose**: Manages document storage, processing, and metadata handling.

**Key Features**:

- **Document Processing**: Text extraction and chunking
- **Metadata Management**: Document classification and tagging
- **Version Control**: Document update tracking
- **Batch Operations**: Efficient bulk processing

**Core Methods**:

```python
class DocumentService:
    async def store_document(self, document: DocumentCreate) -> Document
    async def process_document_chunks(self, document_id: str) -> List[DocumentChunk]
    async def get_document_metadata(self, document_id: str) -> DocumentMetadata
    async def update_document_embeddings(self, document_id: str)
    async def delete_document(self, document_id: str)
```

### 5. Token Tracker (`services/token_tracker.py`)

**Purpose**: Monitors API usage, costs, and performance metrics.

**Key Features**:

- **Real-time Tracking**: Live cost calculation
- **Usage Analytics**: Token consumption patterns
- **Cost Optimization**: Identifies savings opportunities
- **Performance Metrics**: Response time analysis

**Core Methods**:

```python
class TokenTracker:
    def start_request_tracking(self, request_id: str)
    def track_api_call(
        self,
        request_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    )
    def finish_request_tracking(self, request_id: str) -> RequestMetrics
    def get_cost_analysis(self, time_period: str) -> CostAnalysis
```

**Metrics Collected**:

```python
@dataclass
class RequestMetrics:
    request_id: str
    total_cost: float
    total_tokens: int
    response_time: float
    cache_hits: int
    api_calls: List[APICallMetric]
```

### 6. Rate Limiter (`services/rate_limiter.py`)

**Purpose**: Manages API rate limits and implements backoff strategies.

**Key Features**:

- **Adaptive Backoff**: Exponential retry logic
- **Quota Management**: API limit tracking
- **Circuit Breaker**: Prevents cascade failures
- **Priority Queuing**: Important requests first

## 📊 Data Models

### 1. RAG Request Models (`models/rag_models.py`)

**Core Request Structure**:

```python
class RAGRequest(BaseModel):
    query: str = Field(..., description="User's question or query")
    conversation_history: List[ConversationMessage] = Field(
        default=[],
        description="Previous conversation context"
    )
    user_context: Optional[UserContext] = Field(
        default=None,
        description="Additional user context"
    )
    search_config: Optional[SearchConfig] = Field(
        default=None,
        description="Custom search parameters"
    )

class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default=None)

class UserContext(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    domain_expertise: Optional[str] = None
```

**Response Structure**:

```python
class ProcessingResult(BaseModel):
    answer: str = Field(..., description="Generated response")
    sources: List[SourceReference] = Field(
        default=[],
        description="Supporting sources"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Response confidence (0-1)"
    )
    processing_metadata: ProcessingMetadata = Field(
        ...,
        description="Processing details"
    )

class SourceReference(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    content_preview: str
    relevance_score: float
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ProcessingMetadata(BaseModel):
    total_cost: float
    processing_time: float
    tokens_used: int
    cache_hit: bool
    model_used: str
    stages_completed: List[str]
```

### 2. Document Models (`models/document_models.py`)

**Document Structure**:

```python
class Document(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentMetadata(BaseModel):
    source_type: str = Field(..., description="Document source type")
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    tags: List[str] = Field(default=[])
    category: Optional[str] = None
    language: str = Field(default="en")
    word_count: Optional[int] = None

class DocumentChunk(BaseModel):
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    embedding: Optional[List[float]] = Field(default=None)
    chunk_index: int = Field(..., description="Position in document")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")

class ChunkMetadata(BaseModel):
    start_char: int
    end_char: int
    token_count: int
    section_title: Optional[str] = None
    importance_score: Optional[float] = None
```

### 3. Agent Models (`models/agent_models.py`)

**Agent Configuration**:

```python
class AgentConfig(BaseModel):
    agent_type: AgentType = Field(..., description="Type of agent")
    model: str = Field(..., description="AI model to use")
    temperature: float = Field(default=0.1, description="Model temperature")
    max_tokens: int = Field(default=1000, description="Max response tokens")
    system_prompt: str = Field(..., description="Agent system prompt")

class AgentType(str, Enum):
    QUERY_REWRITER = "query_rewriter"
    CONTEXT_DECISION = "context_decision"
    SOURCE_RETRIEVAL = "source_retrieval"
    ANSWER_GENERATION = "answer_generation"

class AgentResponse(BaseModel):
    agent_type: AgentType
    result: Any = Field(..., description="Agent-specific result")
    confidence: Optional[float] = Field(default=None)
    reasoning: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
```

## 🔄 Service Interactions

### Dependency Flow

```
FastAPI Endpoint
    ↓ (Dependency Injection)
OptimizedRAGOrchestrator
    ↓ (Uses)
┌─────────────────┬─────────────────┬─────────────────┐
│   OpenAI        │   Vector        │   Cache         │
│   Service       │   Search        │   Service       │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                    ↓                    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   Token         │   Document      │   Rate          │
│   Tracker       │   Service       │   Limiter       │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                    ↓                    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   OpenAI API    │   Supabase      │   Redis         │
│                 │   Vector DB     │   Cache         │
└─────────────────┴─────────────────┴─────────────────┘
```

### Service Communication Patterns

1. **Orchestrator → Services**: Direct method calls with dependency injection
2. **Services → External APIs**: Async HTTP calls with retry logic
3. **Services → Database**: Connection pooling with transaction management
4. **Services → Cache**: In-memory and Redis-based caching
5. **Cross-Service**: Event-driven updates for cache invalidation

## 🔧 Configuration Management

### Environment Configuration

```python
class RAGConfig(BaseModel):
    # OpenAI Configuration
    openai_api_key: str
    openai_model_primary: str = "gpt-3.5-turbo"
    openai_model_fallback: str = "gpt-3.5-turbo"
    openai_embedding_model: str = "text-embedding-ada-002"

    # Supabase Configuration
    supabase_url: str
    supabase_key: str

    # Cache Configuration
    redis_url: Optional[str] = None
    cache_ttl_responses: int = 86400  # 24 hours
    cache_ttl_embeddings: int = 604800  # 7 days

    # Search Configuration
    similarity_threshold: float = 0.7
    max_search_results: int = 10
    recency_boost_factor: float = 0.1

    # Performance Configuration
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
```

### Service Initialization

```python
# dependencies.py
@lru_cache()
def get_openai_service() -> OpenAIService:
    return OpenAIService(
        api_key=settings.openai_api_key,
        default_model=settings.openai_model_primary
    )

@lru_cache()
def get_vector_search_service() -> VectorSearchService:
    return VectorSearchService(
        supabase_client=get_supabase_client(),
        openai_service=get_openai_service()
    )

@lru_cache()
def get_optimized_orchestrator() -> OptimizedRAGPipelineOrchestrator:
    return OptimizedRAGPipelineOrchestrator(
        agent_registry=get_agent_registry(),
        openai_service=get_openai_service(),
        vector_search_service=get_vector_search_service(),
        cache_service=get_cache_service(),
        token_tracker=get_token_tracker()
    )
```

## 📈 Performance Characteristics

### Service Performance Metrics

| Service          | Avg Response Time | Cache Hit Rate | Cost per Request |
| ---------------- | ----------------- | -------------- | ---------------- |
| OpenAI Service   | 1.2s              | 85%            | $0.002-0.08      |
| Vector Search    | 0.3s              | 70%            | $0.0004          |
| Cache Service    | 0.01s             | 95%            | $0.00001         |
| Document Service | 0.5s              | 60%            | $0.001           |

### Optimization Strategies

1. **Caching Layers**:

   - L1: In-memory cache (fastest)
   - L2: Redis cache (fast)
   - L3: Database cache (slower)

2. **Connection Pooling**:

   - Supabase: 10 concurrent connections
   - Redis: 5 concurrent connections
   - OpenAI: Rate-limited connections

3. **Async Processing**:

   - Non-blocking I/O operations
   - Concurrent API calls where possible
   - Background task processing

4. **Resource Management**:
   - Automatic connection cleanup
   - Memory usage monitoring
   - Graceful degradation under load

## 🚨 Error Handling Strategies

### Service-Level Error Handling

1. **OpenAI Service**:

   - Rate limit handling with exponential backoff
   - Model fallback (GPT-4 → GPT-3.5)
   - Timeout handling with retries

2. **Vector Search Service**:

   - Database connection failures
   - Embedding generation failures
   - Search timeout handling

3. **Cache Service**:
   - Redis connection failures
   - Memory overflow handling
   - Cache corruption recovery

### Global Error Handling

```python
class RAGErrorHandler:
    async def handle_service_error(
        self,
        error: Exception,
        service: str,
        context: Dict
    ) -> ErrorResponse:

        if isinstance(error, OpenAIError):
            return await self.handle_openai_error(error, context)
        elif isinstance(error, DatabaseError):
            return await self.handle_database_error(error, context)
        elif isinstance(error, CacheError):
            return await self.handle_cache_error(error, context)
        else:
            return await self.handle_generic_error(error, context)
```

This comprehensive documentation provides developers with a complete understanding of the RAG system's services and models, enabling effective development, debugging, and optimization.

# RAG System Technical Implementation Guide

## 🎯 Overview

This guide provides comprehensive technical information for developers working with the RAG (Retrieval-Augmented Generation) system. It covers setup, configuration, development patterns, and best practices for extending and maintaining the system.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- FastAPI
- Supabase account with vector database enabled
- OpenAI API key
- Redis (optional, for enhanced caching)

### Environment Setup

```bash
# Clone and setup
git clone <repository>
cd backend

# Install dependencies
pip install -r requirements.txt

# Environment configuration
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Required Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

# Optional: Redis for enhanced caching
REDIS_URL=redis://localhost:6379

# Optional: Custom model configurations
OPENAI_MODEL_PRIMARY=gpt-3.5-turbo
OPENAI_MODEL_FALLBACK=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

## 🏗️ System Architecture Deep Dive

### Core Pipeline Components

#### 1. API Layer (`api/v1/rag_pipeline.py`)

The entry point for all RAG requests, providing both synchronous and streaming endpoints.

```python
# Main processing endpoint
@router.post("/process", response_model=ProcessingResult)
async def process_rag_query(
    request: RAGRequest,
    use_full_pipeline: bool = Query(False),
    orchestrator: Any = Depends(get_optimized_orchestrator),
    full_orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> ProcessingResult:
    """
    Process a RAG query with optional pipeline selection.

    Args:
        request: RAG request containing query and context
        use_full_pipeline: Whether to use full or optimized pipeline
        orchestrator: Injected optimized orchestrator
        full_orchestrator: Injected full orchestrator

    Returns:
        ProcessingResult with answer, sources, and metadata
    """
```

**Key Features**:

- **Pipeline Selection**: Choose between optimized (cost-efficient) and full (maximum accuracy) pipelines
- **Dependency Injection**: Automatic service injection via FastAPI dependencies
- **Error Handling**: Comprehensive error catching and response formatting
- **Streaming Support**: Real-time response streaming for better UX

#### 2. Orchestration Layer (`core/rag_pipeline_optimized.py`)

The brain of the system, coordinating all RAG stages with intelligent optimizations.

```python
class OptimizedRAGPipelineOrchestrator:
    def __init__(
        self,
        agent_registry: AgentRegistry,
        openai_service: OpenAIService,
        vector_search_service: VectorSearchService,
        cache_service: CacheService,
        token_tracker: TokenTracker
    ):
        """Initialize orchestrator with all required services."""

    async def process_query(self, request: RAGRequest) -> ProcessingResult:
        """
        Main processing method with optimization stages:
        1. Cache check (saves ~$0.08 per hit)
        2. Pattern matching (saves ~$0.06 for simple queries)
        3. Context decision (smart retrieval)
        4. Source retrieval (when needed)
        5. Answer generation (optimized model selection)
        """
```

**Optimization Strategies**:

- **Aggressive Caching**: 24-hour TTL for responses, 7-day TTL for embeddings
- **Pattern Matching**: Immediate responses for greetings, thanks, etc.
- **Smart Model Selection**: GPT-3.5 for simple queries, GPT-4 for complex ones
- **Context Decision**: AI-powered decision on whether retrieval is needed

#### 3. Agent System (`core/agents/`)

Specialized AI agents handle different aspects of the RAG process:

```python
class ContextDecisionAgent:
    """Determines if external context is needed for a query."""

    async def process(
        self,
        query: str,
        conversation_history: List[ConversationMessage]
    ) -> ContextDecisionResult:
        """
        Analyze query to determine if retrieval is necessary.

        Returns:
            ContextDecisionResult with decision and confidence score
        """

class SourceRetrievalAgent:
    """Handles semantic search and source ranking."""

    async def process(
        self,
        query: str,
        search_config: SearchConfig
    ) -> List[SourceReference]:
        """
        Perform semantic search and return ranked sources.

        Returns:
            List of relevant sources with relevance scores
        """

class AnswerGenerationAgent:
    """Generates final responses using retrieved context."""

    async def process(
        self,
        query: str,
        sources: List[SourceReference],
        conversation_history: List[ConversationMessage]
    ) -> AnswerGenerationResult:
        """
        Generate contextual response with citations.

        Returns:
            AnswerGenerationResult with response and source citations
        """
```

## 🔧 Development Patterns

### 1. Adding New Agents

To add a new agent to the system:

```python
# 1. Create agent class
class CustomAgent:
    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service
        self.system_prompt = "Your custom system prompt here"

    async def process(self, input_data: Any) -> CustomResult:
        # Implement your agent logic
        pass

# 2. Register in agent registry
class AgentRegistry:
    def __init__(self):
        self.agents = {
            "custom_agent": CustomAgent,
            # ... other agents
        }

    def get_agent(self, agent_type: str) -> Any:
        return self.agents.get(agent_type)

# 3. Use in orchestrator
async def process_with_custom_agent(self, request: RAGRequest):
    custom_agent = self.agent_registry.get_agent("custom_agent")
    result = await custom_agent.process(request.query)
    return result
```

### 2. Implementing Custom Caching

```python
class CustomCacheService:
    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis_client = redis_client
        self.memory_cache = {}

    async def get_cached_item(self, key: str) -> Optional[Any]:
        # Try memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Try Redis cache
        if self.redis_client:
            cached = await self.redis_client.get(key)
            if cached:
                return json.loads(cached)

        return None

    async def cache_item(self, key: str, value: Any, ttl: int = 3600):
        # Cache in memory
        self.memory_cache[key] = value

        # Cache in Redis with TTL
        if self.redis_client:
            await self.redis_client.setex(
                key,
                ttl,
                json.dumps(value, default=str)
            )
```

### 3. Custom Vector Search Implementation

```python
class CustomVectorSearchService:
    def __init__(
        self,
        supabase_client: Client,
        openai_service: OpenAIService
    ):
        self.supabase = supabase_client
        self.openai_service = openai_service

    async def semantic_search(
        self,
        query: str,
        config: SearchConfig
    ) -> List[SearchResult]:
        # Generate query embedding
        embedding = await self.openai_service.create_embedding(query)

        # Perform vector search
        results = await self.supabase.rpc(
            'search_embeddings',
            {
                'query_embedding': embedding,
                'similarity_threshold': config.similarity_threshold,
                'match_count': config.max_results
            }
        ).execute()

        # Post-process results
        return self.post_process_results(results.data, config)

    def post_process_results(
        self,
        raw_results: List[Dict],
        config: SearchConfig
    ) -> List[SearchResult]:
        processed = []
        for result in raw_results:
            # Apply recency boost
            recency_score = self.calculate_recency_boost(
                result['created_at'],
                config.recency_boost_factor
            )

            # Combine similarity and recency
            final_score = result['similarity'] + recency_score

            processed.append(SearchResult(
                document_id=result['document_id'],
                chunk_id=result['chunk_id'],
                content=result['content'],
                relevance_score=final_score,
                metadata=result.get('metadata', {})
            ))

        # Sort by final score
        return sorted(processed, key=lambda x: x.relevance_score, reverse=True)
```

## 📊 Performance Optimization

### 1. Caching Strategies

#### Response Caching

```python
class ResponseCacheManager:
    def __init__(self):
        self.cache_ttl = {
            'responses': 86400,  # 24 hours
            'embeddings': 604800,  # 7 days
            'patterns': 2592000,  # 30 days
        }

    def generate_cache_key(self, request: RAGRequest) -> str:
        """Generate deterministic cache key from request."""
        key_data = {
            'query': request.query.lower().strip(),
            'history_hash': self.hash_conversation_history(
                request.conversation_history
            ),
            'context_hash': self.hash_user_context(request.user_context)
        }
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()

    async def get_or_compute(
        self,
        request: RAGRequest,
        compute_func: Callable
    ) -> ProcessingResult:
        cache_key = self.generate_cache_key(request)

        # Try cache first
        cached_result = await self.get_cached_response(cache_key)
        if cached_result:
            return cached_result

        # Compute and cache
        result = await compute_func(request)
        await self.cache_response(cache_key, result)
        return result
```

#### Embedding Caching

```python
class EmbeddingCacheManager:
    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service
        self.embedding_cache = {}

    async def get_or_create_embedding(self, text: str) -> List[float]:
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Check cache
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        # Create and cache
        embedding = await self.openai_service.create_embedding(text)
        self.embedding_cache[text_hash] = embedding
        return embedding
```

### 2. Database Optimization

#### Connection Pooling

```python
class DatabaseManager:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client = create_client(
            supabase_url,
            supabase_key,
            options=ClientOptions(
                postgrest_client_timeout=10,
                storage_client_timeout=10,
                schema="public",
                auto_refresh_token=True,
                persist_session=True,
            )
        )

    async def execute_with_retry(
        self,
        operation: Callable,
        max_retries: int = 3
    ):
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

#### Query Optimization

```python
class OptimizedQueries:
    @staticmethod
    def build_vector_search_query(
        embedding: List[float],
        config: SearchConfig
    ) -> Dict:
        """Build optimized vector search query."""
        return {
            'query_embedding': embedding,
            'similarity_threshold': config.similarity_threshold,
            'match_count': config.max_results,
            'include_metadata': config.include_metadata,
            # Use indexes for better performance
            'use_index': True,
            'index_type': 'ivfflat'
        }

    @staticmethod
    def build_document_filter(
        filters: Dict[str, Any]
    ) -> str:
        """Build SQL filter for document queries."""
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(f"{key} IN ({','.join(map(str, value))})")
            else:
                conditions.append(f"{key} = '{value}'")
        return " AND ".join(conditions)
```

## 🔍 Monitoring and Debugging

### 1. Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'request_count': 0,
            'total_cost': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0
        }

    @contextmanager
    def track_request(self, request_id: str):
        start_time = time.time()
        try:
            yield
            self.metrics['request_count'] += 1
        finally:
            duration = time.time() - start_time
            self.update_avg_response_time(duration)

    def update_avg_response_time(self, duration: float):
        current_avg = self.metrics['avg_response_time']
        count = self.metrics['request_count']
        self.metrics['avg_response_time'] = (
            (current_avg * (count - 1) + duration) / count
        )

    def get_performance_report(self) -> Dict:
        return {
            **self.metrics,
            'cache_hit_rate': (
                self.metrics['cache_hits'] /
                (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0
                else 0
            )
        }
```

### 2. Logging and Debugging

```python
import logging
from typing import Any, Dict

class RAGLogger:
    def __init__(self, name: str = "rag_system"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_request(self, request_id: str, request: RAGRequest):
        self.logger.info(
            f"Request {request_id}: Query='{request.query[:100]}...'"
        )

    def log_stage_completion(
        self,
        request_id: str,
        stage: str,
        duration: float,
        metadata: Dict[str, Any] = None
    ):
        self.logger.info(
            f"Request {request_id}: Stage '{stage}' completed in {duration:.2f}s"
        )
        if metadata:
            self.logger.debug(f"Stage metadata: {metadata}")

    def log_error(
        self,
        request_id: str,
        error: Exception,
        context: Dict[str, Any] = None
    ):
        self.logger.error(
            f"Request {request_id}: Error - {str(error)}"
        )
        if context:
            self.logger.error(f"Error context: {context}")
```

## 🧪 Testing Strategies

### 1. Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.core.rag_pipeline_optimized import OptimizedRAGPipelineOrchestrator

class TestOptimizedRAGOrchestrator:
    @pytest.fixture
    def mock_services(self):
        return {
            'agent_registry': MagicMock(),
            'openai_service': AsyncMock(),
            'vector_search_service': AsyncMock(),
            'cache_service': AsyncMock(),
            'token_tracker': MagicMock()
        }

    @pytest.fixture
    def orchestrator(self, mock_services):
        return OptimizedRAGPipelineOrchestrator(**mock_services)

    @pytest.mark.asyncio
    async def test_cache_hit_scenario(self, orchestrator, mock_services):
        # Setup
        request = RAGRequest(query="test query")
        cached_result = ProcessingResult(
            answer="cached answer",
            sources=[],
            processing_metadata=ProcessingMetadata(
                total_cost=0.0,
                processing_time=0.01,
                tokens_used=0,
                cache_hit=True,
                model_used="cache",
                stages_completed=["cache"]
            )
        )
        mock_services['cache_service'].get_cached_response.return_value = cached_result

        # Execute
        result = await orchestrator.process_query(request)

        # Assert
        assert result.answer == "cached answer"
        assert result.processing_metadata.cache_hit is True
        assert result.processing_metadata.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_pattern_matching_scenario(self, orchestrator, mock_services):
        # Setup
        request = RAGRequest(query="hello")
        mock_services['cache_service'].get_cached_response.return_value = None

        # Execute
        result = await orchestrator.process_query(request)

        # Assert
        assert "hello" in result.answer.lower()
        assert result.processing_metadata.total_cost < 0.01  # Pattern response cost
```

### 2. Integration Testing

```python
class TestRAGIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_rag_flow(self):
        # Setup real services (with test configuration)
        orchestrator = await self.setup_test_orchestrator()

        # Test query
        request = RAGRequest(
            query="What is machine learning?",
            conversation_history=[]
        )

        # Execute
        result = await orchestrator.process_query(request)

        # Assert
        assert result.answer is not None
        assert len(result.answer) > 50  # Substantial response
        assert result.processing_metadata.total_cost > 0
        assert len(result.sources) >= 0  # May or may not have sources

    async def setup_test_orchestrator(self):
        # Initialize with test configuration
        test_config = {
            'openai_api_key': os.getenv('TEST_OPENAI_API_KEY'),
            'supabase_url': os.getenv('TEST_SUPABASE_URL'),
            'supabase_key': os.getenv('TEST_SUPABASE_KEY'),
        }

        # Create test services
        openai_service = OpenAIService(test_config['openai_api_key'])
        # ... initialize other services

        return OptimizedRAGPipelineOrchestrator(
            # ... pass test services
        )
```

## 🚀 Deployment Considerations

### 1. Environment Configuration

```python
class DeploymentConfig:
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        base_config = {
            'openai_model_primary': 'gpt-3.5-turbo',
            'cache_ttl_responses': 86400,
            'max_concurrent_requests': 10,
        }

        if self.environment == "production":
            return {
                **base_config,
                'openai_model_primary': 'gpt-4',
                'cache_ttl_responses': 86400,
                'max_concurrent_requests': 50,
                'enable_monitoring': True,
            }
        elif self.environment == "staging":
            return {
                **base_config,
                'cache_ttl_responses': 3600,
                'max_concurrent_requests': 20,
                'enable_monitoring': True,
            }
        else:  # development
            return {
                **base_config,
                'cache_ttl_responses': 300,
                'max_concurrent_requests': 5,
                'enable_monitoring': False,
            }
```

### 2. Health Checks

```python
class HealthChecker:
    def __init__(self, orchestrator: OptimizedRAGPipelineOrchestrator):
        self.orchestrator = orchestrator

    async def check_system_health(self) -> Dict[str, Any]:
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.utcnow().isoformat()
        }

        # Check OpenAI service
        try:
            await self.orchestrator.openai_service.create_embedding("test")
            health_status['checks']['openai'] = 'healthy'
        except Exception as e:
            health_status['checks']['openai'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'

        # Check Supabase connection
        try:
            await self.orchestrator.vector_search_service.supabase.table('documents').select('id').limit(1).execute()
            health_status['checks']['supabase'] = 'healthy'
        except Exception as e:
            health_status['checks']['supabase'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'

        # Check cache service
        try:
            await self.orchestrator.cache_service.get_cached_response('health_check')
            health_status['checks']['cache'] = 'healthy'
        except Exception as e:
            health_status['checks']['cache'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'

        return health_status
```

## 📚 Best Practices

### 1. Error Handling

- Always use try-catch blocks for external API calls
- Implement exponential backoff for retries
- Provide meaningful error messages to users
- Log errors with sufficient context for debugging

### 2. Performance

- Cache aggressively but invalidate intelligently
- Use connection pooling for database operations
- Implement rate limiting to prevent abuse
- Monitor token usage and costs continuously

### 3. Security

- Validate all input data
- Sanitize user queries before processing
- Use environment variables for sensitive configuration
- Implement proper authentication and authorization

### 4. Maintainability

- Follow consistent coding patterns
- Document all public interfaces
- Use type hints throughout the codebase
- Write comprehensive tests for critical paths

This technical implementation guide provides the foundation for working effectively with the RAG system, enabling developers to understand, extend, and maintain the codebase efficiently.

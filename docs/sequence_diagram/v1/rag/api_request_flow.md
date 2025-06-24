# RAG API Request Flow Documentation

## 🎯 Overview

This document details the complete request flow from the FastAPI endpoint through the RAG pipeline to the final response. It covers both the optimized and full pipeline variants, dependency injection, and error handling.

## 🚀 API Endpoints Analysis

### Primary Endpoint: `/api/v1/rag/process`

**File**: `backend/app/api/v1/rag_pipeline.py`

```python
@router.post("/process", response_model=ProcessingResult)
async def process_rag_query(
    request: RAGRequest,
    use_full_pipeline: bool = Query(False),
    orchestrator: Any = Depends(get_optimized_orchestrator),
    full_orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> ProcessingResult:
```

#### Request Flow Breakdown

1. **Request Validation**

   - FastAPI validates `RAGRequest` model
   - Pydantic ensures data integrity
   - Query parameters processed

2. **Dependency Injection**

   - `get_optimized_orchestrator()` → `OptimizedRAGPipelineOrchestrator`
   - `get_pipeline_orchestrator()` → `RAGPipelineOrchestrator`
   - Dependencies cached via `@lru_cache()`

3. **Pipeline Selection Logic**
   ```python
   if use_full_pipeline:
       # Use full orchestrator for maximum accuracy
       result = await full_orchestrator.process_query(...)
   else:
       # Use optimized orchestrator (default)
       processing_result = await orchestrator.process_query(request)
   ```

## 🔄 Request Processing Flow

### Step 1: Request Initialization

```python
# Generate unique request ID
request_id = str(uuid.uuid4())

# Start token tracking for cost analysis
token_tracker.start_request_tracking(request_id, request.query, "optimized")
```

### Step 2: Cache Check (Optimized Pipeline)

```python
if self.enable_aggressive_caching:
    cache_key = self._generate_cache_key(request.query, request.conversation_history)
    cached_result = self._get_cached_result(cache_key)
    if cached_result:
        logger.info(f"Cache hit for request {request_id} - saved ~$0.08")
        return cached_result
```

**Cache Key Generation**:

```python
def _generate_cache_key(self, query: str, conversation_history: Optional[List]) -> str:
    query_hash = hashlib.md5(query.lower().encode()).hexdigest()
    if conversation_history:
        history_str = json.dumps(conversation_history[-2:], sort_keys=True)
        history_hash = hashlib.md5(history_str.encode()).hexdigest()
        return f"{query_hash}_{history_hash}"
    return query_hash
```

### Step 3: Pattern Matching (Optimized Pipeline)

```python
if self.enable_pattern_matching:
    pattern_result = self._handle_non_informational_patterns(request.query)
    if pattern_result:
        # Handle greetings, thanks, etc. without AI calls
        return ProcessingResult(...)
```

**Pattern Examples**:

- Greetings: `hello`, `hi`, `good morning`
- Thanks: `thank you`, `thanks`, `terima kasih`
- Farewells: `bye`, `goodbye`, `see you`

### Step 4: Model Selection (Optimized Pipeline)

```python
use_advanced_model = self._should_use_advanced_model(request.query, request.conversation_history)

# Criteria for advanced model (GPT-4):
# - Complex analytical questions
# - Multi-step reasoning
# - Conversation context exists
# - Technical explanations needed
```

### Step 5: Pipeline Processing

Both pipelines follow the same 4-stage process:

#### Stage 1: Query Rewriting

```python
# Enhanced query preprocessing (optimized) or full rewriting (full)
rewritten_query = await self._enhanced_query_rewriting(query, stage_results, request_id)
```

#### Stage 2: Context Decision

```python
# Determine if external context retrieval is needed
context_needed = await self._smart_context_decision(
    rewritten_query, conversation_history, stage_results, request_id
)
```

#### Stage 3: Source Retrieval (Conditional)

```python
sources = []
if context_needed:
    sources = await self._quality_source_retrieval(rewritten_query, stage_results, request_id)
```

#### Stage 4: Answer Generation

```python
# Generate final response with smart model selection
final_response = await self._smart_answer_generation(
    rewritten_query, sources, stage_results, request_id, use_advanced_model
)
```

## 🏗️ Dependency Injection Deep Dive

### Core Dependencies (`dependencies.py`)

```python
@lru_cache()
def get_supabase_client() -> Client:
    """Supabase client for vector database operations"""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

@lru_cache()
def get_openai_service() -> OpenAIService:
    """OpenAI service for LLM and embedding operations"""
    return OpenAIService()

@lru_cache()
def get_agent_registry() -> AgentRegistry:
    """Registry for managing RAG agents"""
    registry = AgentRegistry()
    # Register agent types
    registry.register_agent_type("query_rewriter", QueryRewritingAgent)
    registry.register_agent_type("context_decision", ContextDecisionAgent)
    registry.register_agent_type("source_retrieval", SourceRetrievalAgent)
    registry.register_agent_type("answer_generation", AnswerGenerationAgent)
    return registry
```

### Pipeline Orchestrator Dependencies

```python
def get_pipeline_orchestrator(
    agent_registry=Depends(get_agent_registry),
    agent_metrics=Depends(get_agent_metrics)
) -> RAGPipelineOrchestrator:
    """Full pipeline orchestrator with all features"""
    global _pipeline_orchestrator
    if _pipeline_orchestrator is None:
        _pipeline_orchestrator = RAGPipelineOrchestrator(
            agent_registry=agent_registry,
            agent_metrics=agent_metrics,
            config={
                "max_pipeline_duration": 30.0,
                "enable_fallbacks": True,
                "enable_caching": True,
                "enable_streaming": True
            }
        )
    return _pipeline_orchestrator

def get_optimized_orchestrator(
    agent_registry=Depends(get_agent_registry),
    agent_metrics=Depends(get_agent_metrics)
) -> Any:
    """Optimized pipeline orchestrator for cost efficiency"""
    global _optimized_orchestrator
    if _optimized_orchestrator is None:
        from ...core.rag_pipeline_optimized import OptimizedRAGPipelineOrchestrator
        _optimized_orchestrator = OptimizedRAGPipelineOrchestrator()
    return _optimized_orchestrator
```

## 📊 Request/Response Models

### Input Model: `RAGRequest`

```python
class RAGRequest(BaseAPIModel):
    query: str = Field(..., description="User query", min_length=1, max_length=2000)
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None)
    user_context: Optional[Dict[str, Any]] = Field(None)
    pipeline_config: Optional[Dict[str, Any]] = Field(None)
```

**Example Request**:

```json
{
  "query": "What are the benefits of using microservices architecture?",
  "conversation_history": [
    { "role": "user", "content": "Tell me about software architecture" },
    { "role": "assistant", "content": "Software architecture defines..." }
  ],
  "user_context": {
    "domain": "software_engineering",
    "experience_level": "intermediate"
  },
  "pipeline_config": {
    "max_sources": 5,
    "similarity_threshold": 0.8
  }
}
```

### Output Model: `ProcessingResult`

```python
class ProcessingResult(BaseAPIModel):
    request_id: str                    # Unique identifier
    query: str                         # Original query
    status: str                        # "completed", "failed"
    pipeline_type: str                 # "optimized", "full"
    final_response: Optional[Dict]     # Generated response
    stage_results: Dict                # Results from each stage
    total_duration: float              # Processing time in seconds
    optimization_info: Optional[Dict]  # Cost and optimization data
```

**Example Response**:

```json
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "query": "What are the benefits of using microservices architecture?",
  "status": "completed",
  "pipeline_type": "optimized",
  "final_response": {
    "query": "What are the benefits of using microservices architecture?",
    "response": {
      "content": "Microservices architecture offers several key benefits:\n\n1. **Scalability**...",
      "citations": [
        {
          "source_id": "doc_123",
          "title": "Microservices Patterns",
          "relevance_score": 0.92
        }
      ],
      "format_type": "markdown"
    }
  },
  "stage_results": {
    "context_decision": {
      "context_needed": true,
      "confidence": 0.85,
      "reasoning": "Technical query requires documentation context"
    },
    "source_retrieval": {
      "sources_count": 3,
      "strategy": "quality_hybrid",
      "duration": 0.234
    },
    "answer_generation": {
      "model_used": "gpt-4-turbo",
      "duration": 1.456,
      "optimization": "smart_model_selection"
    }
  },
  "total_duration": 2.1,
  "optimization_info": {
    "pipeline_used": "quality_optimized",
    "model_used": "gpt-4-turbo",
    "optimization": "balanced"
  }
}
```

## 🔄 Alternative Endpoints

### Full Pipeline Endpoint: `/api/v1/rag/process/full`

```python
@router.post("/process/full", response_model=ProcessingResult)
async def process_rag_query_full(
    request: RAGRequest,
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
) -> ProcessingResult:
    """Always uses the full pipeline for maximum accuracy"""
```

### Streaming Endpoint: `/api/v1/rag/stream`

```python
@router.post("/stream")
async def stream_query(
    request: RAGStreamRequest,
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """Server-Sent Events streaming"""
    async def generate_stream():
        async for update in orchestrator.stream_query(...):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")
```

### WebSocket Endpoint: `/api/v1/rag/stream-ws`

```python
@router.websocket("/stream-ws")
async def websocket_stream_query(
    websocket: WebSocket,
    orchestrator: RAGPipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """Real-time WebSocket streaming"""
    await websocket.accept()
    # Handle real-time communication
```

## ⚠️ Error Handling

### Error Response Format

```python
# Standardized error response
return ProcessingResult(
    request_id=request_id,
    query=request.query,
    status="failed",
    pipeline_type="optimized",
    final_response={
        "query": request.query,
        "response": {
            "content": f"I apologize, but I encountered an error: {error_msg}",
            "citations": [],
            "format_type": "markdown"
        }
    },
    stage_results={"error": error_msg},
    total_duration=0.0,
    optimization_info={"pipeline_used": "optimized", "error": True}
)
```

### Common Error Scenarios

1. **OpenAI API Failures**

   - Rate limiting
   - API timeouts
   - Invalid API keys

2. **Database Errors**

   - Supabase connection issues
   - Vector search failures
   - Embedding storage problems

3. **Pipeline Errors**
   - Agent initialization failures
   - Processing timeouts
   - Memory limitations

## 📈 Performance Monitoring

### Token Tracking

```python
# Start tracking
token_tracker.start_request_tracking(request_id, request.query, "optimized")

# Track individual API calls
token_tracker.track_api_call(
    request_id=request_id,
    call_type="answer_generation",
    model="gpt-4-turbo",
    prompt_tokens=1500,
    completion_tokens=300,
    prompt_text=prompt_text,
    completion_text=response_text
)

# Finish tracking
token_tracker.finish_request_tracking(request_id)
```

### Metrics Collection

- **Request Duration**: Total processing time
- **Stage Performance**: Individual stage timings
- **Cache Hit Rates**: Optimization effectiveness
- **Token Usage**: Cost tracking and optimization
- **Error Rates**: System reliability metrics

## 🔗 Integration Points

### External Services

- **OpenAI API**: LLM and embedding services
- **Supabase**: Vector database and storage
- **Redis**: Caching layer (optional)

### Internal Services

- **Agent Registry**: Agent lifecycle management
- **Vector Search**: Document retrieval
- **Document Service**: Content management
- **Rate Limiter**: API throttling

This request flow documentation provides the foundation for understanding how requests move through the RAG system. For implementation details of each stage, see the [Pipeline Processing Details](./pipeline_processing.md).

# Document Processing Services and Models Documentation

## 🎯 Overview

This document provides comprehensive coverage of all services and data models used in the document processing system. It details service architecture, data flow patterns, model schemas, and integration points between components.

## 🏗️ Core Services Architecture

### 1. Document Processor Service

**Purpose**: Core text extraction and chunking service for multi-format document processing.

**File Location**: `backend/app/services/document_processor.py`

#### Service Interface

```python
class DocumentProcessor:
    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service
        self.encoding = tiktoken.get_encoding("cl100k_base")

    async def validate_file(self, file_content: bytes, filename: str) -> Tuple[bool, str]:
        """Validate file content and format support"""

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[Dict] = None
    ) -> ProcessingResult:
        """Main document processing pipeline"""
```

#### Processing Pipeline Methods

**Text Extraction Methods**:

```python
async def _extract_text_pdf(self, file_content: bytes) -> str:
    """Extract text from PDF files using PyPDF2"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text_content = ""

    for page_num, page in enumerate(pdf_reader.pages):
        try:
            page_text = page.extract_text()
            if page_text.strip():
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")

    return text_content.strip()

async def _extract_text_docx(self, file_content: bytes) -> str:
    """Extract text from DOCX files using python-docx"""
    doc = Document(io.BytesIO(file_content))
    paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return "\n".join(paragraphs)

async def _extract_text_html(self, file_content: bytes) -> str:
    """Extract text from HTML files using BeautifulSoup"""
    soup = BeautifulSoup(file_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text and clean whitespace
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)

async def _extract_text_plain(self, file_content: bytes) -> str:
    """Extract text from plain text files with encoding detection"""
    encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            return file_content.decode(encoding)
        except UnicodeDecodeError:
            continue

    # Fallback: decode with errors='ignore'
    return file_content.decode('utf-8', errors='ignore')
```

**Chunking Strategy**:

```python
async def _create_chunks(
    self,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    filename: str
) -> List[DocumentChunk]:
    """Intelligent text chunking with sentence boundary preservation"""

    # Use NLTK for sentence tokenization
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    chunk_index = 0
    start_sentence = 0

    for sentence_idx, sentence in enumerate(sentences):
        sentence_tokens = len(self.encoding.encode(sentence))

        # Check if adding sentence would exceed chunk size
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Create chunk with metadata
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                chunk_index=chunk_index,
                token_count=current_tokens,
                metadata={
                    "filename": filename,
                    "start_sentence": start_sentence,
                    "end_sentence": sentence_idx - 1,
                    "total_sentences": sentence_idx - start_sentence,
                    "processing_timestamp": datetime.utcnow().isoformat()
                }
            )
            chunks.append(chunk)

            # Calculate overlap for next chunk
            overlap_text = self._calculate_overlap(current_chunk, chunk_overlap)
            current_chunk = overlap_text + " " + sentence
            current_tokens = len(self.encoding.encode(current_chunk))
            chunk_index += 1
            start_sentence = sentence_idx
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens

    # Handle remaining content
    if current_chunk.strip():
        chunk = DocumentChunk(
            text=current_chunk.strip(),
            chunk_index=chunk_index,
            token_count=current_tokens,
            metadata={
                "filename": filename,
                "start_sentence": start_sentence,
                "end_sentence": len(sentences) - 1,
                "total_sentences": len(sentences) - start_sentence,
                "processing_timestamp": datetime.utcnow().isoformat()
            }
        )
        chunks.append(chunk)

    return chunks

def _calculate_overlap(self, text: str, overlap_tokens: int) -> str:
    """Calculate text overlap for chunk boundaries"""
    if overlap_tokens <= 0:
        return ""

    tokens = self.encoding.encode(text)
    if len(tokens) <= overlap_tokens:
        return text

    overlap_token_slice = tokens[-overlap_tokens:]
    return self.encoding.decode(overlap_token_slice)
```

**Embedding Integration**:

```python
async def _generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Generate embeddings for document chunks using OpenAI service"""
    batch_size = 100  # OpenAI batch size limit

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk.text for chunk in batch]

        try:
            embeddings = await self.openai_service.create_embeddings_batch(texts)

            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding

        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
            raise

    return chunks
```

#### Performance Metrics

```python
class ProcessingMetrics:
    """Track processing performance metrics"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.stage_times: Dict[str, float] = {}

    def start_processing(self):
        self.start_time = time.time()

    def record_stage(self, stage_name: str, duration: float):
        self.stage_times[stage_name] = duration

    def get_total_time(self) -> float:
        return sum(self.stage_times.values()) if self.stage_times else 0

    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            "total_processing_time": self.get_total_time(),
            "stage_breakdown": self.stage_times,
            "throughput_metrics": {
                "avg_extraction_speed": self.stage_times.get("text_extraction", 0),
                "chunking_efficiency": self.stage_times.get("chunking", 0),
                "embedding_generation_time": self.stage_times.get("embedding_generation", 0)
            }
        }
```

### 2. Document Service

**Purpose**: Database operations, document management, and search functionality.

**File Location**: `backend/app/services/document_service.py`

#### Service Interface

```python
class DocumentService:
    def __init__(self, supabase_client):
        self.supabase = supabase_client

    async def store_document(
        self,
        processing_result: ProcessingResult,
        user_id: Optional[str] = None
    ) -> DocumentResponse:
        """Store processed document and embeddings in database"""

    async def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        search_query: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> DocumentListResponse:
        """List documents with pagination and filtering"""

    async def search_documents(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform semantic search using vector embeddings"""
```

#### Database Operations

**Document Storage**:

```python
async def store_document(
    self,
    processing_result: ProcessingResult,
    user_id: Optional[str] = None
) -> DocumentResponse:
    """Store document with optimized batch operations"""

    document_id = str(uuid.uuid4())

    # Prepare document record
    document_data = {
        "id": document_id,
        "title": processing_result.filename,
        "content": processing_result.full_text[:50000],  # Truncate for storage
        "metadata": {
            **processing_result.metadata,
            "chunk_count": len(processing_result.chunks),
            "total_tokens": sum(chunk.token_count for chunk in processing_result.chunks),
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

    # Insert document
    document_result = self.supabase.table("documents").insert(document_data).execute()

    if not document_result.data:
        raise ValueError("Failed to store document")

    # Prepare embeddings for batch insert
    embeddings_data = []
    for chunk in processing_result.chunks:
        embedding_record = {
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "chunk_text": chunk.text,
            "embedding": chunk.embedding,
            "chunk_index": chunk.chunk_index,
            "chunk_metadata": {
                **chunk.metadata,
                "token_count": chunk.token_count
            },
            "created_at": datetime.utcnow().isoformat()
        }
        embeddings_data.append(embedding_record)

    # Batch insert embeddings
    if embeddings_data:
        embeddings_result = self.supabase.table("embeddings").insert(embeddings_data).execute()

        if not embeddings_result.data:
            # Rollback document if embedding storage fails
            self.supabase.table("documents").delete().eq("id", document_id).execute()
            raise ValueError("Failed to store document embeddings")

    return DocumentResponse(
        id=document_id,
        filename=processing_result.filename,
        content_type=processing_result.metadata.get("content_type"),
        file_size=processing_result.metadata.get("file_size"),
        chunk_count=len(processing_result.chunks),
        created_at=document_data["created_at"],
        metadata=document_data["metadata"],
        processing_stats={
            "total_chunks": len(processing_result.chunks),
            "total_tokens": sum(chunk.token_count for chunk in processing_result.chunks),
            "avg_chunk_size": sum(chunk.token_count for chunk in processing_result.chunks) / len(processing_result.chunks),
            "processing_time": processing_result.metadata.get("processing_time", 0)
        }
    )
```

**Vector Search Operations**:

```python
async def search_documents(
    self,
    query_embedding: List[float],
    limit: int = 10,
    threshold: float = 0.7,
    document_ids: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    metadata_filters: Optional[Dict] = None
) -> List[SearchResult]:
    """Advanced semantic search with filtering and ranking"""

    # Build base query using Supabase RPC function
    rpc_params = {
        "query_embedding": query_embedding,
        "match_threshold": threshold,
        "match_count": limit * 2  # Get more results for post-processing
    }

    # Add optional filters
    if document_ids:
        rpc_params["document_ids"] = document_ids
    if user_id:
        rpc_params["user_id"] = user_id

    # Execute vector search
    search_result = self.supabase.rpc("search_embeddings_with_metadata", rpc_params).execute()

    if not search_result.data:
        return []

    # Post-process results
    processed_results = []
    for result in search_result.data:
        # Apply metadata filters
        if metadata_filters and not self._matches_metadata_filters(result, metadata_filters):
            continue

        # Apply recency boost if configured
        similarity_score = result["similarity"]
        if result.get("created_at"):
            similarity_score = self._apply_recency_boost(similarity_score, result["created_at"])

        search_result_obj = SearchResult(
            chunk_id=result["chunk_id"],
            document_id=result["document_id"],
            filename=result["filename"],
            chunk_text=result["chunk_text"],
            similarity=similarity_score,
            chunk_index=result["chunk_index"],
            metadata=result.get("chunk_metadata", {})
        )
        processed_results.append(search_result_obj)

    # Sort by similarity and limit results
    processed_results.sort(key=lambda x: x.similarity, reverse=True)
    return processed_results[:limit]

def _apply_recency_boost(self, similarity: float, created_at: str, boost_factor: float = 0.1) -> float:
    """Apply recency boost to search results"""
    try:
        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        days_old = (datetime.utcnow().replace(tzinfo=timezone.utc) - created_date).days

        # Apply exponential decay for recency
        recency_multiplier = math.exp(-days_old / 30)  # 30-day half-life
        return similarity + (boost_factor * recency_multiplier)
    except:
        return similarity

def _matches_metadata_filters(self, result: Dict, filters: Dict) -> bool:
    """Check if result matches metadata filters"""
    result_metadata = result.get("chunk_metadata", {})

    for filter_key, filter_value in filters.items():
        if filter_key not in result_metadata:
            return False

        result_value = result_metadata[filter_key]

        # Handle different filter types
        if isinstance(filter_value, list):
            if result_value not in filter_value:
                return False
        elif isinstance(filter_value, str):
            if filter_value.lower() not in str(result_value).lower():
                return False
        elif result_value != filter_value:
            return False

    return True
```

**Analytics and Statistics**:

```python
async def get_document_statistics(self, user_id: Optional[str] = None) -> DocumentStats:
    """Comprehensive document processing statistics"""

    # Document count query
    doc_query = self.supabase.table("documents").select("id", count="exact")
    if user_id:
        doc_query = doc_query.eq("user_id", user_id)
    doc_result = doc_query.execute()
    total_documents = doc_result.count or 0

    # Chunk and token analysis
    chunk_query = self.supabase.table("embeddings").select("chunk_metadata", count="exact")
    if user_id:
        chunk_query = chunk_query.in_(
            "document_id",
            self.supabase.table("documents").select("id").eq("user_id", user_id)
        )
    chunk_result = chunk_query.execute()

    total_chunks = chunk_result.count or 0
    total_tokens = 0
    chunk_sizes = []

    for chunk_data in chunk_result.data:
        metadata = chunk_data.get("chunk_metadata", {})
        token_count = metadata.get("token_count", 0)
        total_tokens += token_count
        if token_count > 0:
            chunk_sizes.append(token_count)

    # Calculate processing efficiency metrics
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    processing_efficiency = self._calculate_processing_efficiency(total_documents, total_chunks, total_tokens)

    return DocumentStats(
        total_documents=total_documents,
        total_chunks=total_chunks,
        total_tokens=total_tokens,
        avg_chunk_size=avg_chunk_size,
        supported_formats=["pdf", "docx", "txt", "html", "md"],
        processing_stats={
            "avg_processing_time": processing_efficiency.get("avg_processing_time", 0),
            "success_rate": processing_efficiency.get("success_rate", 0),
            "throughput_per_hour": processing_efficiency.get("throughput_per_hour", 0)
        }
    )
```

### 3. OpenAI Service Integration

**Purpose**: Embedding generation and AI model interactions with caching and rate limiting.

**File Location**: `backend/app/services/openai_service.py`

#### Embedding Operations

```python
class OpenAIService:
    def __init__(self, api_key: str, cache_service: CacheService, rate_limiter: RateLimiter):
        self.client = OpenAI(api_key=api_key)
        self.cache = cache_service
        self.rate_limiter = rate_limiter

    async def create_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """Generate embeddings with caching and rate limiting"""

        # Check cache for existing embeddings
        cached_embeddings = {}
        uncached_texts = []
        text_indices = {}

        for i, text in enumerate(texts):
            cache_key = f"embedding:{model}:{hashlib.md5(text.encode()).hexdigest()}"
            cached_result = await self.cache.get(cache_key)

            if cached_result:
                cached_embeddings[i] = cached_result
            else:
                uncached_texts.append(text)
                text_indices[len(uncached_texts) - 1] = i

        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            # Apply rate limiting
            await self.rate_limiter.acquire("openai_embeddings", len(uncached_texts))

            try:
                response = await self.client.embeddings.create(
                    input=uncached_texts,
                    model=model
                )

                new_embeddings = [data.embedding for data in response.data]

                # Cache new embeddings
                for i, embedding in enumerate(new_embeddings):
                    original_index = text_indices[i]
                    text = texts[original_index]
                    cache_key = f"embedding:{model}:{hashlib.md5(text.encode()).hexdigest()}"
                    await self.cache.set(cache_key, embedding, ttl=86400 * 7)  # 7 days

            except Exception as e:
                logger.error(f"OpenAI embedding generation failed: {e}")
                raise

        # Combine cached and new embeddings in original order
        result_embeddings = [None] * len(texts)

        # Place cached embeddings
        for i, embedding in cached_embeddings.items():
            result_embeddings[i] = embedding

        # Place new embeddings
        new_embedding_index = 0
        for i in range(len(texts)):
            if result_embeddings[i] is None:
                result_embeddings[i] = new_embeddings[new_embedding_index]
                new_embedding_index += 1

        return result_embeddings
```

## 📊 Data Models and Schemas

### Document Processing Models

#### Core Data Models

```python
class DocumentChunk(BaseModel):
    """Individual text chunk with metadata and embedding"""
    text: str
    chunk_index: int
    token_count: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}

class ProcessingResult(BaseModel):
    """Complete document processing result"""
    filename: str
    full_text: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    processing_metrics: Dict[str, float] = {}

class DocumentResponse(BaseModel):
    """API response for document operations"""
    id: str
    filename: str
    content_type: Optional[str]
    file_size: Optional[int]
    chunk_count: int
    created_at: str
    metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]

class BatchUploadResponse(BaseModel):
    """Response for batch upload operations"""
    successful_uploads: List[DocumentResponse]
    failed_uploads: List[Dict[str, str]]
    total_processed: int
    success_count: int
    failure_count: int
    processing_time_seconds: float
```

#### Search Models

```python
class SearchRequest(BaseModel):
    """Search request parameters"""
    query: str
    max_results: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = None
    user_id: Optional[str] = None
    boost_recent: bool = False
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    metadata_filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    """Individual search result"""
    chunk_id: str
    document_id: str
    filename: str
    chunk_text: str
    similarity: float
    chunk_index: int
    metadata: Dict[str, Any] = {}

class SearchResponse(BaseModel):
    """Complete search response"""
    query: str
    search_type: str = "semantic"
    results: List[SearchResult]
    total_results: int
    filtered_results: int
    query_time: float
    avg_similarity: float
    metadata: Dict[str, Any] = {}
```

#### Statistics and Analytics Models

```python
class DocumentStats(BaseModel):
    """Document processing statistics"""
    total_documents: int
    total_chunks: int
    total_tokens: int
    avg_chunk_size: float
    supported_formats: List[str]
    processing_stats: Dict[str, float]

class ProcessingMetrics(BaseModel):
    """Detailed processing performance metrics"""
    total_processing_time: float
    stage_breakdown: Dict[str, float]
    throughput_metrics: Dict[str, float]
    cost_analysis: Dict[str, float] = {}
    efficiency_scores: Dict[str, float] = {}
```

### Database Schema

#### Documents Table

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX idx_documents_title ON documents USING gin(to_tsvector('english', title));
CREATE INDEX idx_documents_metadata ON documents USING gin(metadata);
```

#### Embeddings Table

```sql
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1536),
    chunk_index INTEGER NOT NULL,
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for vector search and performance
CREATE INDEX idx_embeddings_document_id ON embeddings(document_id);
CREATE INDEX idx_embeddings_chunk_index ON embeddings(chunk_index);
CREATE INDEX idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_embeddings_metadata ON embeddings USING gin(chunk_metadata);
```

#### Supabase RPC Functions

```sql
-- Vector search function with metadata filtering
CREATE OR REPLACE FUNCTION search_embeddings_with_metadata(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10,
    document_ids uuid[] DEFAULT NULL,
    user_id uuid DEFAULT NULL
)
RETURNS TABLE (
    chunk_id uuid,
    document_id uuid,
    filename text,
    chunk_text text,
    similarity float,
    chunk_index integer,
    chunk_metadata jsonb,
    created_at timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id as chunk_id,
        e.document_id,
        d.title as filename,
        e.chunk_text,
        1 - (e.embedding <=> query_embedding) as similarity,
        e.chunk_index,
        e.chunk_metadata,
        e.created_at
    FROM embeddings e
    JOIN documents d ON e.document_id = d.id
    WHERE
        (1 - (e.embedding <=> query_embedding)) > match_threshold
        AND (document_ids IS NULL OR e.document_id = ANY(document_ids))
        AND (user_id IS NULL OR d.user_id = user_id)
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;
```

## 🔄 Service Integration Patterns

### Dependency Injection

```python
# backend/app/core/dependencies.py
from functools import lru_cache

@lru_cache()
def get_document_processor() -> DocumentProcessor:
    """Get cached document processor instance"""
    openai_service = get_openai_service()
    return DocumentProcessor(openai_service)

@lru_cache()
def get_document_service() -> DocumentService:
    """Get cached document service instance"""
    supabase_client = get_supabase_client()
    return DocumentService(supabase_client)

@lru_cache()
def get_openai_service() -> OpenAIService:
    """Get cached OpenAI service instance"""
    cache_service = get_cache_service()
    rate_limiter = get_rate_limiter()
    return OpenAIService(
        api_key=settings.OPENAI_API_KEY,
        cache_service=cache_service,
        rate_limiter=rate_limiter
    )
```

### Error Handling Strategy

```python
class DocumentProcessingError(Exception):
    """Base exception for document processing errors"""
    def __init__(self, message: str, stage: str, details: Dict = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}

class TextExtractionError(DocumentProcessingError):
    """Error during text extraction stage"""
    pass

class ChunkingError(DocumentProcessingError):
    """Error during text chunking stage"""
    pass

class EmbeddingError(DocumentProcessingError):
    """Error during embedding generation"""
    pass

class StorageError(DocumentProcessingError):
    """Error during database storage"""
    pass

# Error handling wrapper
async def handle_processing_errors(func, *args, **kwargs):
    """Centralized error handling for processing operations"""
    try:
        return await func(*args, **kwargs)
    except DocumentProcessingError as e:
        logger.error(f"Processing error in {e.stage}: {e}", extra={"details": e.details})
        raise HTTPException(
            status_code=422,
            detail={
                "error": str(e),
                "stage": e.stage,
                "details": e.details
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")
```

## 📈 Performance Optimization

### Caching Strategy

```python
class CacheService:
    """Multi-level caching for document processing"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache = {}

    async def get_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding with fallback layers"""
        # Level 1: In-memory cache
        if text_hash in self.local_cache:
            return self.local_cache[text_hash]

        # Level 2: Redis cache
        cached_data = await self.redis.get(f"embedding:{text_hash}")
        if cached_data:
            embedding = json.loads(cached_data)
            self.local_cache[text_hash] = embedding
            return embedding

        return None

    async def cache_embedding(self, text_hash: str, embedding: List[float], ttl: int = 86400 * 7):
        """Cache embedding at multiple levels"""
        # Store in Redis with TTL
        await self.redis.setex(
            f"embedding:{text_hash}",
            ttl,
            json.dumps(embedding)
        )

        # Store in local cache (with size limit)
        if len(self.local_cache) < 1000:
            self.local_cache[text_hash] = embedding
```

### Database Optimization

```python
class OptimizedDocumentService(DocumentService):
    """Enhanced document service with performance optimizations"""

    async def bulk_store_embeddings(self, embeddings_data: List[Dict]) -> bool:
        """Optimized bulk embedding storage"""
        batch_size = 1000

        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i:i + batch_size]

            # Use prepared statement for better performance
            result = await self.supabase.table("embeddings").insert(batch).execute()

            if not result.data:
                logger.error(f"Failed to insert embedding batch {i//batch_size + 1}")
                return False

        return True

    async def optimized_vector_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        **filters
    ) -> List[SearchResult]:
        """Optimized vector search with smart indexing"""

        # Use approximate nearest neighbor search for large datasets
        search_params = {
            "query_embedding": query_embedding,
            "match_count": limit,
            "probes": 10,  # IVF index parameter
            **filters
        }

        result = await self.supabase.rpc(
            "optimized_vector_search",
            search_params
        ).execute()

        return [SearchResult(**item) for item in result.data]
```

This comprehensive documentation provides developers with detailed understanding of all services and models in the document processing system, enabling effective development and integration work.

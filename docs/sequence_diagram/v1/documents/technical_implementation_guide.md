# Document Processing Technical Implementation Guide

## 🎯 Overview

This guide provides comprehensive implementation details for developers working with the Document Processing system. It covers setup, development patterns, customization options, performance optimization, testing strategies, and deployment considerations.

## 🚀 Quick Start Setup

### Prerequisites

```bash
# System Requirements
- Python 3.8+
- PostgreSQL 12+ with pgvector extension
- Redis (optional, for caching)
- OpenAI API key
- Supabase account and project

# Required Python packages
pip install fastapi uvicorn supabase python-multipart
pip install PyPDF2 python-docx beautifulsoup4 nltk
pip install openai tiktoken
pip install python-magic-bin  # Windows
pip install python-magic      # Linux/Mac
```

### Environment Configuration

```bash
# .env file configuration
OPENAI_API_KEY=your_openai_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_service_role_key

# Optional configurations
CHUNK_SIZE_DEFAULT=1000
CHUNK_OVERLAP_DEFAULT=200
MAX_FILE_SIZE_MB=50
REDIS_URL=redis://localhost:6379/0

# Performance settings
MAX_CONCURRENT_UPLOADS=5
PROCESSING_TIMEOUT=300
EMBEDDING_BATCH_SIZE=100
```

### Database Setup

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create embeddings table
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1536),
    chunk_index INTEGER NOT NULL,
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX idx_embeddings_document_id ON embeddings(document_id);
CREATE INDEX idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops);
```

### Development Server

```bash
# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# With debug logging
LOG_LEVEL=DEBUG uvicorn app.main:app --reload

# Test the API
curl http://localhost:8000/api/v1/documents/formats/supported
```

## 🏗️ System Architecture Deep Dive

### Service Layer Architecture

```python
# Service composition pattern
class DocumentProcessingOrchestrator:
    """Main orchestrator for document processing operations"""

    def __init__(
        self,
        document_processor: DocumentProcessor,
        document_service: DocumentService,
        cache_service: CacheService,
        rate_limiter: RateLimiter
    ):
        self.document_processor = document_processor
        self.document_service = document_service
        self.cache = cache_service
        self.rate_limiter = rate_limiter

    async def process_document_pipeline(
        self,
        file_content: bytes,
        filename: str,
        processing_options: ProcessingOptions
    ) -> ProcessingResult:
        """Complete document processing pipeline with error handling"""

        # Stage 1: Validation and preprocessing
        await self._validate_and_preprocess(file_content, filename)

        # Stage 2: Text extraction with fallback strategies
        extracted_text = await self._extract_text_with_fallback(
            file_content, filename, processing_options
        )

        # Stage 3: Intelligent chunking
        chunks = await self._create_optimized_chunks(
            extracted_text, processing_options
        )

        # Stage 4: Embedding generation with batching
        chunks_with_embeddings = await self._generate_embeddings_batch(chunks)

        # Stage 5: Storage with transaction management
        result = await self._store_with_transaction(
            chunks_with_embeddings, processing_options
        )

        return result
```

### Configuration Management

```python
# settings.py - Centralized configuration
from pydantic import BaseSettings
from typing import List, Optional

class DocumentProcessingSettings(BaseSettings):
    """Document processing configuration"""

    # File processing
    chunk_size_default: int = 1000
    chunk_size_min: int = 100
    chunk_size_max: int = 4000
    chunk_overlap_default: int = 200
    chunk_overlap_max: int = 1000
    max_file_size_mb: int = 50

    # Supported formats
    supported_mime_types: List[str] = [
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/html",
        "text/markdown"
    ]

    # Performance settings
    max_concurrent_uploads: int = 5
    processing_timeout: int = 300
    embedding_batch_size: int = 100

    # OpenAI settings
    openai_model: str = "text-embedding-ada-002"
    openai_max_retries: int = 3
    openai_timeout: int = 30

    # Cache settings
    cache_embedding_ttl: int = 86400 * 7  # 7 days
    cache_response_ttl: int = 86400      # 1 day

    class Config:
        env_prefix = "DOC_PROCESSING_"

settings = DocumentProcessingSettings()
```

## 🔧 Development Patterns

### Custom Document Processor Extension

```python
# Custom processor for specialized formats
class CustomDocumentProcessor(DocumentProcessor):
    """Extended processor with custom format support"""

    def __init__(self, openai_service: OpenAIService):
        super().__init__(openai_service)
        self.custom_extractors = {
            "application/vnd.ms-excel": self._extract_text_excel,
            "application/vnd.ms-powerpoint": self._extract_text_powerpoint,
            "text/csv": self._extract_text_csv
        }

    async def _extract_text_excel(self, file_content: bytes) -> str:
        """Extract text from Excel files"""
        import pandas as pd

        try:
            # Read Excel file
            df = pd.read_excel(io.BytesIO(file_content))

            # Convert to text representation
            text_content = []
            for sheet_name, sheet_data in df.items():
                text_content.append(f"Sheet: {sheet_name}")
                text_content.append(sheet_data.to_string(index=False))

            return "\n\n".join(text_content)

        except Exception as e:
            logger.error(f"Excel extraction failed: {e}")
            raise TextExtractionError(f"Failed to extract Excel content: {e}")

    async def _extract_text_csv(self, file_content: bytes) -> str:
        """Extract text from CSV files with intelligent formatting"""
        import csv

        try:
            content = file_content.decode('utf-8')
            csv_reader = csv.reader(io.StringIO(content))

            # Convert CSV to readable text format
            rows = list(csv_reader)
            if not rows:
                return ""

            # Format with headers
            headers = rows[0]
            formatted_rows = [f"Headers: {', '.join(headers)}"]

            for row in rows[1:]:
                row_text = []
                for header, value in zip(headers, row):
                    if value.strip():
                        row_text.append(f"{header}: {value}")
                formatted_rows.append(" | ".join(row_text))

            return "\n".join(formatted_rows)

        except Exception as e:
            logger.error(f"CSV extraction failed: {e}")
            raise TextExtractionError(f"Failed to extract CSV content: {e}")
```

### Smart Chunking Strategies

```python
class AdvancedChunkingStrategy:
    """Advanced chunking with semantic boundary detection"""

    def __init__(self, encoding):
        self.encoding = encoding
        self.semantic_separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "? ",    # Question endings
            "! ",    # Exclamation endings
        ]

    async def create_semantic_chunks(
        self,
        text: str,
        target_size: int,
        overlap: int,
        preserve_paragraphs: bool = True
    ) -> List[DocumentChunk]:
        """Create chunks with semantic boundary preservation"""

        if preserve_paragraphs:
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            return await self._chunk_by_paragraphs(paragraphs, target_size, overlap)
        else:
            # Use sentence-based chunking
            sentences = sent_tokenize(text)
            return await self._chunk_by_sentences(sentences, target_size, overlap)

    async def _chunk_by_paragraphs(
        self,
        paragraphs: List[str],
        target_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """Paragraph-aware chunking"""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        paragraph_buffer = []

        for paragraph in paragraphs:
            para_tokens = len(self.encoding.encode(paragraph))

            # If single paragraph exceeds target, break it down
            if para_tokens > target_size:
                # Process current buffer first
                if paragraph_buffer:
                    chunk = await self._finalize_chunk(paragraph_buffer, len(chunks))
                    chunks.append(chunk)
                    paragraph_buffer = []
                    current_tokens = 0

                # Break down large paragraph
                sub_chunks = await self._break_large_paragraph(paragraph, target_size)
                chunks.extend(sub_chunks)

            elif current_tokens + para_tokens > target_size:
                # Create chunk from buffer
                if paragraph_buffer:
                    chunk = await self._finalize_chunk(paragraph_buffer, len(chunks))
                    chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._calculate_paragraph_overlap(paragraph_buffer, overlap)
                paragraph_buffer = [overlap_text, paragraph] if overlap_text else [paragraph]
                current_tokens = len(self.encoding.encode(" ".join(paragraph_buffer)))

            else:
                paragraph_buffer.append(paragraph)
                current_tokens += para_tokens

        # Handle remaining paragraphs
        if paragraph_buffer:
            chunk = await self._finalize_chunk(paragraph_buffer, len(chunks))
            chunks.append(chunk)

        return chunks
```

### Error Handling and Recovery

```python
class RobustDocumentProcessor:
    """Document processor with comprehensive error handling"""

    def __init__(self, base_processor: DocumentProcessor):
        self.base_processor = base_processor
        self.fallback_strategies = {
            "pdf": [self._fallback_pdf_ocr, self._fallback_pdf_text],
            "docx": [self._fallback_docx_xml, self._fallback_docx_text],
            "general": [self._fallback_charset_detection]
        }

    async def process_with_recovery(
        self,
        file_content: bytes,
        filename: str,
        **kwargs
    ) -> ProcessingResult:
        """Process document with automatic error recovery"""

        try:
            # Primary processing attempt
            return await self.base_processor.process_document(
                file_content, filename, **kwargs
            )

        except TextExtractionError as e:
            logger.warning(f"Primary extraction failed: {e}, trying fallbacks")
            return await self._attempt_fallback_extraction(file_content, filename, **kwargs)

        except EmbeddingError as e:
            logger.warning(f"Embedding generation failed: {e}, trying alternative approach")
            return await self._handle_embedding_failure(file_content, filename, **kwargs)

        except Exception as e:
            logger.error(f"Unexpected processing error: {e}")
            return await self._create_minimal_result(file_content, filename, str(e))

    async def _attempt_fallback_extraction(
        self,
        file_content: bytes,
        filename: str,
        **kwargs
    ) -> ProcessingResult:
        """Try fallback extraction strategies"""

        file_extension = filename.split('.')[-1].lower()
        strategies = self.fallback_strategies.get(file_extension, self.fallback_strategies["general"])

        for strategy in strategies:
            try:
                extracted_text = await strategy(file_content, filename)
                if extracted_text and len(extracted_text.strip()) > 50:
                    # Process with extracted text
                    return await self._process_extracted_text(
                        extracted_text, filename, **kwargs
                    )
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy.__name__} failed: {e}")
                continue

        # All strategies failed
        raise TextExtractionError(
            "All text extraction strategies failed",
            "text_extraction",
            {"filename": filename, "attempted_strategies": len(strategies)}
        )
```

## 📊 Performance Optimization

### Batch Processing Optimization

```python
class OptimizedBatchProcessor:
    """High-performance batch document processing"""

    def __init__(
        self,
        processor: DocumentProcessor,
        max_concurrent: int = 5,
        chunk_batch_size: int = 100
    ):
        self.processor = processor
        self.max_concurrent = max_concurrent
        self.chunk_batch_size = chunk_batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch_optimized(
        self,
        files: List[UploadFile],
        processing_options: ProcessingOptions
    ) -> BatchResult:
        """Process multiple files with optimized resource usage"""

        # Group files by type for optimized processing
        file_groups = self._group_files_by_type(files)

        # Process groups concurrently
        all_results = []
        for file_type, file_list in file_groups.items():
            group_results = await self._process_file_group(
                file_list, file_type, processing_options
            )
            all_results.extend(group_results)

        return self._compile_batch_results(all_results)

    async def _process_file_group(
        self,
        files: List[UploadFile],
        file_type: str,
        options: ProcessingOptions
    ) -> List[ProcessingResult]:
        """Process files of same type with type-specific optimizations"""

        # Create optimized processor for this file type
        specialized_processor = self._get_specialized_processor(file_type)

        # Process files concurrently within the group
        tasks = []
        for file in files:
            task = self._process_single_file_safe(
                file, specialized_processor, options
            )
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_single_file_safe(
        self,
        file: UploadFile,
        processor: DocumentProcessor,
        options: ProcessingOptions
    ) -> ProcessingResult:
        """Process single file with resource management"""

        async with self.semaphore:
            try:
                file_content = await file.read()
                return await processor.process_document(
                    file_content, file.filename, **options.dict()
                )
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {e}")
                return ProcessingError(filename=file.filename, error=str(e))
```

### Database Performance Optimization

```python
class OptimizedDocumentService:
    """Document service with performance optimizations"""

    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.connection_pool = None
        self._init_connection_pool()

    async def bulk_store_optimized(
        self,
        processing_results: List[ProcessingResult],
        user_id: Optional[str] = None
    ) -> List[DocumentResponse]:
        """Optimized bulk storage with transaction management"""

        # Prepare all data for batch operations
        documents_data, embeddings_data = self._prepare_bulk_data(
            processing_results, user_id
        )

        # Use database transaction for consistency
        async with self._get_transaction() as txn:
            try:
                # Bulk insert documents
                document_results = await self._bulk_insert_documents(
                    documents_data, txn
                )

                # Bulk insert embeddings in chunks
                await self._bulk_insert_embeddings_chunked(
                    embeddings_data, txn
                )

                await txn.commit()
                return document_results

            except Exception as e:
                await txn.rollback()
                logger.error(f"Bulk storage transaction failed: {e}")
                raise

    async def _bulk_insert_embeddings_chunked(
        self,
        embeddings_data: List[Dict],
        transaction
    ):
        """Insert embeddings in optimized chunks"""

        chunk_size = 1000  # Optimal batch size for Supabase

        for i in range(0, len(embeddings_data), chunk_size):
            chunk = embeddings_data[i:i + chunk_size]

            result = await transaction.table("embeddings").insert(chunk).execute()

            if not result.data:
                raise ValueError(f"Failed to insert embedding chunk {i//chunk_size + 1}")

            # Brief pause to prevent overwhelming the database
            if i + chunk_size < len(embeddings_data):
                await asyncio.sleep(0.01)
```

### Caching Strategy Implementation

```python
class MultiLevelCacheService:
    """Advanced caching with multiple storage layers"""

    def __init__(self, redis_client, local_cache_size: int = 1000):
        self.redis = redis_client
        self.local_cache = LRUCache(maxsize=local_cache_size)
        self.compression_enabled = True

    async def get_embedding_cached(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> Optional[List[float]]:
        """Multi-level embedding retrieval with compression"""

        # Generate cache key
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = f"embedding:{model}:{text_hash}"

        # Level 1: Local memory cache
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]

        # Level 2: Redis cache with compression
        cached_data = await self._get_from_redis_compressed(cache_key)
        if cached_data:
            # Store in local cache for faster access
            self.local_cache[cache_key] = cached_data
            return cached_data

        return None

    async def cache_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str = "text-embedding-ada-002",
        ttl: int = 86400 * 7
    ):
        """Multi-level embedding caching with compression"""

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = f"embedding:{model}:{text_hash}"

        # Store in local cache
        self.local_cache[cache_key] = embedding

        # Store in Redis with compression
        await self._store_in_redis_compressed(cache_key, embedding, ttl)

    async def _get_from_redis_compressed(self, key: str) -> Optional[List[float]]:
        """Retrieve and decompress data from Redis"""
        try:
            compressed_data = await self.redis.get(key)
            if compressed_data:
                if self.compression_enabled:
                    decompressed = gzip.decompress(compressed_data)
                    return json.loads(decompressed.decode())
                else:
                    return json.loads(compressed_data)
        except Exception as e:
            logger.warning(f"Redis cache retrieval failed: {e}")
        return None

    async def _store_in_redis_compressed(
        self,
        key: str,
        data: List[float],
        ttl: int
    ):
        """Compress and store data in Redis"""
        try:
            json_data = json.dumps(data)

            if self.compression_enabled:
                compressed_data = gzip.compress(json_data.encode())
                await self.redis.setex(key, ttl, compressed_data)
            else:
                await self.redis.setex(key, ttl, json_data)

        except Exception as e:
            logger.warning(f"Redis cache storage failed: {e}")
```

## 🧪 Testing Strategies

### Unit Testing

```python
import pytest
from unittest.mock import Mock, AsyncMock
from app.services.document_processor import DocumentProcessor

class TestDocumentProcessor:
    """Comprehensive unit tests for DocumentProcessor"""

    @pytest.fixture
    def mock_openai_service(self):
        """Mock OpenAI service for testing"""
        service = Mock()
        service.create_embeddings_batch = AsyncMock(
            return_value=[[0.1] * 1536] * 5  # Mock embeddings
        )
        return service

    @pytest.fixture
    def document_processor(self, mock_openai_service):
        """Document processor with mocked dependencies"""
        return DocumentProcessor(mock_openai_service)

    @pytest.mark.asyncio
    async def test_pdf_text_extraction(self, document_processor):
        """Test PDF text extraction functionality"""
        # Load test PDF content
        with open("tests/fixtures/test_document.pdf", "rb") as f:
            pdf_content = f.read()

        # Extract text
        extracted_text = await document_processor._extract_text_pdf(pdf_content)

        # Assertions
        assert len(extracted_text) > 0
        assert "test content" in extracted_text.lower()
        assert "--- Page 1 ---" in extracted_text

    @pytest.mark.asyncio
    async def test_chunking_strategy(self, document_processor):
        """Test intelligent chunking with various scenarios"""

        test_cases = [
            {
                "text": "Short text.",
                "chunk_size": 1000,
                "expected_chunks": 1
            },
            {
                "text": "A" * 5000,  # Very long text
                "chunk_size": 1000,
                "expected_chunks": 5
            },
            {
                "text": ". ".join(["Sentence"] * 100),  # Many sentences
                "chunk_size": 500,
                "expected_chunks": lambda x: x >= 2
            }
        ]

        for case in test_cases:
            chunks = await document_processor._create_chunks(
                case["text"],
                case["chunk_size"],
                200,  # overlap
                "test.txt"
            )

            if callable(case["expected_chunks"]):
                assert case["expected_chunks"](len(chunks))
            else:
                assert len(chunks) == case["expected_chunks"]

    @pytest.mark.asyncio
    async def test_processing_pipeline_error_handling(self, document_processor):
        """Test error handling in processing pipeline"""

        # Test with corrupted file content
        with pytest.raises(TextExtractionError):
            await document_processor.process_document(
                b"invalid content",
                "test.pdf",
                "application/pdf"
            )
```

### Integration Testing

```python
@pytest.mark.integration
class TestDocumentProcessingIntegration:
    """Integration tests for complete document processing flow"""

    @pytest.fixture(scope="class")
    async def test_client(self):
        """Test client with real database connection"""
        from app.main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_complete_upload_flow(self, test_client):
        """Test complete document upload and processing"""

        # Prepare test file
        test_content = b"This is test document content for integration testing."

        # Upload document
        files = {"file": ("test.txt", test_content, "text/plain")}
        data = {"chunk_size": 500, "chunk_overlap": 100}

        response = await test_client.post(
            "/api/v1/documents/upload",
            files=files,
            data=data
        )

        # Verify response
        assert response.status_code == 201
        result = response.json()
        assert "id" in result
        assert result["filename"] == "test.txt"
        assert result["chunk_count"] > 0

        # Verify document can be retrieved
        document_id = result["id"]
        get_response = await test_client.get(f"/api/v1/documents/{document_id}")
        assert get_response.status_code == 200

        # Verify search functionality
        search_data = {
            "query": "test document content",
            "max_results": 5
        }
        search_response = await test_client.post(
            "/api/v1/documents/search",
            json=search_data
        )
        assert search_response.status_code == 200
        search_results = search_response.json()
        assert len(search_results["results"]) > 0

    @pytest.mark.asyncio
    async def test_batch_upload_performance(self, test_client):
        """Test batch upload performance and reliability"""

        # Create multiple test files
        test_files = []
        for i in range(10):
            content = f"Test document {i} with unique content for testing batch upload functionality."
            test_files.append(("files", (f"test_{i}.txt", content.encode(), "text/plain")))

        # Upload batch
        data = {"chunk_size": 300}
        response = await test_client.post(
            "/api/v1/documents/batch-upload",
            files=test_files,
            data=data
        )

        # Verify batch results
        assert response.status_code == 200
        result = response.json()
        assert result["success_count"] == 10
        assert result["failure_count"] == 0
        assert len(result["successful_uploads"]) == 10
```

### Performance Testing

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class PerformanceTestSuite:
    """Performance and load testing for document processing"""

    async def test_concurrent_upload_performance(self):
        """Test system performance under concurrent load"""

        async def upload_single_document(session, document_content, filename):
            """Upload single document and measure performance"""
            start_time = time.time()

            files = {"file": (filename, document_content, "text/plain")}
            data = {"chunk_size": 1000}

            async with session.post("/api/v1/documents/upload", files=files, data=data) as response:
                result = await response.json()
                processing_time = time.time() - start_time

                return {
                    "success": response.status == 201,
                    "processing_time": processing_time,
                    "filename": filename,
                    "chunk_count": result.get("chunk_count", 0)
                }

        # Create test documents
        test_documents = [
            (f"Test document {i} content " * 100, f"test_{i}.txt")
            for i in range(50)
        ]

        # Run concurrent uploads
        async with aiohttp.ClientSession() as session:
            tasks = [
                upload_single_document(session, content, filename)
                for content, filename in test_documents
            ]

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

        # Analyze results
        successful_uploads = [r for r in results if isinstance(r, dict) and r["success"]]
        failed_uploads = [r for r in results if not (isinstance(r, dict) and r.get("success"))]

        avg_processing_time = sum(r["processing_time"] for r in successful_uploads) / len(successful_uploads)
        throughput = len(successful_uploads) / total_time

        print(f"Performance Test Results:")
        print(f"- Total uploads: {len(test_documents)}")
        print(f"- Successful: {len(successful_uploads)}")
        print(f"- Failed: {len(failed_uploads)}")
        print(f"- Average processing time: {avg_processing_time:.2f}s")
        print(f"- Throughput: {throughput:.2f} documents/second")
        print(f"- Total test time: {total_time:.2f}s")

        # Performance assertions
        assert len(successful_uploads) >= len(test_documents) * 0.95  # 95% success rate
        assert avg_processing_time < 5.0  # Average under 5 seconds
        assert throughput > 2.0  # At least 2 documents per second
```

## 🚀 Deployment Considerations

### Production Configuration

```python
# production_settings.py
from app.core.config import DocumentProcessingSettings

class ProductionSettings(DocumentProcessingSettings):
    """Production-optimized configuration"""

    # Performance settings
    max_concurrent_uploads: int = 20
    processing_timeout: int = 600
    embedding_batch_size: int = 200

    # Security settings
    max_file_size_mb: int = 100
    allowed_file_extensions: set = {".pdf", ".docx", ".txt", ".html", ".md"}

    # Database settings
    connection_pool_size: int = 20
    connection_pool_max_overflow: int = 30
    query_timeout: int = 30

    # Cache settings
    redis_connection_pool_size: int = 10
    cache_embedding_ttl: int = 86400 * 30  # 30 days
    local_cache_size: int = 5000

    # Monitoring settings
    enable_performance_metrics: bool = True
    log_level: str = "INFO"
    enable_error_tracking: bool = True
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.8"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  redis_data:
```

### Monitoring and Observability

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Metrics
document_uploads_total = Counter(
    'document_uploads_total',
    'Total number of document uploads',
    ['status', 'file_type']
)

processing_duration = Histogram(
    'document_processing_duration_seconds',
    'Time spent processing documents',
    ['stage']
)

active_processing_jobs = Gauge(
    'active_processing_jobs',
    'Number of currently processing documents'
)

class DocumentProcessingMonitor:
    """Monitoring and metrics collection for document processing"""

    def __init__(self):
        self.logger = structlog.get_logger()

    @contextmanager
    def track_processing_stage(self, stage_name: str):
        """Track processing stage duration"""
        start_time = time.time()
        active_processing_jobs.inc()

        try:
            yield
            processing_duration.labels(stage=stage_name).observe(time.time() - start_time)
            self.logger.info("Processing stage completed", stage=stage_name, duration=time.time() - start_time)
        except Exception as e:
            self.logger.error("Processing stage failed", stage=stage_name, error=str(e))
            raise
        finally:
            active_processing_jobs.dec()

    def record_upload_metric(self, file_type: str, status: str):
        """Record upload metrics"""
        document_uploads_total.labels(status=status, file_type=file_type).inc()
```

This comprehensive technical implementation guide provides developers with all the necessary information to effectively work with, extend, and deploy the Document Processing system.

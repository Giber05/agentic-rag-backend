# Document Processing API Request Flow Documentation

## 🎯 Overview

This document details the complete request flow for all document-related API endpoints, from file upload through processing to storage and retrieval. It covers single uploads, batch processing, search operations, and document management.

## 🚀 API Endpoints Analysis

### Document Upload Endpoints

#### Primary Endpoint: `/api/v1/documents/upload`

**File**: `backend/app/api/v1/documents.py`

```python
@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000, description="Target chunk size in tokens", ge=100, le=4000),
    chunk_overlap: int = Form(200, description="Overlap between chunks in tokens", ge=0, le=1000),
    metadata: Optional[str] = Form(None, description="Additional metadata as JSON string"),
    user_id: Optional[str] = Form(None, description="User ID for ownership"),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    document_service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
```

##### Request Flow Breakdown

1. **File Reception**

   - FastAPI receives multipart/form-data
   - Validates file upload parameters
   - Extracts form fields and file data

2. **Dependency Injection**

   - `get_document_processor()` → `DocumentProcessor`
   - `get_document_service()` → `DocumentService`
   - Dependencies cached via `@lru_cache()`

3. **File Validation**

   ```python
   # Filename validation
   if not file.filename:
       raise HTTPException(status_code=400, detail="Filename is required")

   # File content validation
   file_content = await file.read()
   is_valid, validation_message = await document_processor.validate_file(
       file_content, file.filename
   )
   ```

4. **Metadata Processing**
   ```python
   # Parse optional JSON metadata
   additional_metadata = {}
   if metadata:
       try:
           additional_metadata = json.loads(metadata)
       except json.JSONDecodeError:
           raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
   ```

## 🔄 Document Processing Flow

### Single Document Upload Flow

```python
# Step 1: Document Processing
processing_result = await document_processor.process_document(
    file_content=file_content,
    filename=file.filename,
    content_type=file.content_type,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    metadata=additional_metadata
)

# Step 2: Database Storage
document_response = await document_service.store_document(
    processing_result, user_id
)
```

#### Detailed Processing Stages

##### Stage 1: Content Type Detection

```python
def _detect_content_type(self, file_content: bytes, filename: str) -> str:
    try:
        # Use python-magic for accurate detection
        mime_type = magic.from_buffer(file_content, mime=True)
        if mime_type and mime_type != 'application/octet-stream':
            return mime_type
    except Exception:
        pass

    # Fallback to filename-based detection
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or 'application/octet-stream'
```

##### Stage 2: Text Extraction (Format-Specific)

```python
async def _extract_text(self, file_content: bytes, content_type: str, filename: str) -> str:
    if content_type.startswith('text/'):
        return await self._extract_text_plain(file_content)
    elif content_type == 'application/pdf':
        return await self._extract_text_pdf(file_content)
    elif content_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        return await self._extract_text_docx(file_content)
    elif content_type in ['text/html', 'application/xhtml+xml']:
        return await self._extract_text_html(file_content)
    else:
        # Fallback to plain text
        return await self._extract_text_plain(file_content)
```

##### Stage 3: Intelligent Chunking

```python
async def _create_chunks(
    self,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    filename: str
) -> List[DocumentChunk]:
    # Split text into sentences for better boundaries
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = ""
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = len(self.encoding.encode(sentence))

        # Check if adding sentence exceeds chunk size
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Create chunk
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                chunk_index=chunk_index,
                token_count=current_tokens,
                metadata={
                    "filename": filename,
                    "start_sentence": chunk_start_sentence,
                    "end_sentence": sentence_index - 1
                }
            )
            chunks.append(chunk)

            # Handle overlap
            overlap_text = self._calculate_overlap(current_chunk, chunk_overlap)
            current_chunk = overlap_text + sentence
            current_tokens = len(self.encoding.encode(current_chunk))
            chunk_index += 1
        else:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens

    return chunks
```

##### Stage 4: Embedding Generation

```python
async def _generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    # Process in batches for efficiency
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk.text for chunk in batch]

        # Generate embeddings for batch
        embeddings = await self.openai_service.create_embeddings_batch(texts)

        # Assign embeddings to chunks
        for chunk, embedding in zip(batch, embeddings):
            chunk.embedding = embedding

    return chunks
```

### Batch Upload Endpoint: `/api/v1/documents/batch-upload`

```python
@router.post("/batch-upload", response_model=BatchUploadResponse)
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    user_id: Optional[str] = Form(None),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    document_service: DocumentService = Depends(get_document_service)
):
```

#### Batch Processing Flow

```python
start_time = time.time()
successful_uploads = []
failed_uploads = []

for file in files:
    try:
        # Individual file processing
        file_content = await file.read()

        # Validate file
        is_valid, validation_message = await document_processor.validate_file(
            file_content, file.filename
        )

        if not is_valid:
            failed_uploads.append({
                "filename": file.filename,
                "error": validation_message
            })
            continue

        # Process document
        processing_result = await document_processor.process_document(
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Store in database
        document_response = await document_service.store_document(
            processing_result, user_id
        )
        successful_uploads.append(document_response)

    except Exception as e:
        failed_uploads.append({
            "filename": file.filename or "unknown",
            "error": str(e)
        })

processing_time = time.time() - start_time
```

## 📊 Request/Response Models

### Upload Request (Form Data)

```python
# Multipart form data structure
{
    "file": "Binary file content",
    "chunk_size": 1000,           # Optional: 100-4000
    "chunk_overlap": 200,         # Optional: 0-1000
    "metadata": '{"author": "John Doe", "category": "research"}',  # Optional JSON string
    "user_id": "user_123"         # Optional
}
```

### Upload Response: `DocumentResponse`

```python
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "research_paper.pdf",
    "content_type": "application/pdf",
    "file_size": 2548736,
    "chunk_count": 25,
    "created_at": "2024-01-15T10:30:00Z",
    "metadata": {
        "filename": "research_paper.pdf",
        "content_type": "application/pdf",
        "file_size": 2548736,
        "file_extension": ".pdf",
        "page_count": 12,
        "author": "John Doe",
        "category": "research",
        "processing_timestamp": "2024-01-15T10:30:00Z"
    },
    "processing_stats": {
        "total_chunks": 25,
        "total_tokens": 15420,
        "avg_chunk_size": 616.8,
        "processing_time": 3.2
    }
}
```

### Batch Upload Response: `BatchUploadResponse`

```python
{
    "successful_uploads": [
        {
            "id": "doc1_id",
            "filename": "document1.pdf",
            "chunk_count": 15,
            // ... other DocumentResponse fields
        },
        {
            "id": "doc2_id",
            "filename": "document2.docx",
            "chunk_count": 8,
            // ... other DocumentResponse fields
        }
    ],
    "failed_uploads": [
        {
            "filename": "corrupted_file.pdf",
            "error": "File validation failed: Unable to read PDF content"
        }
    ],
    "total_processed": 3,
    "success_count": 2,
    "failure_count": 1,
    "processing_time_seconds": 8.7
}
```

## 🔍 Document Retrieval and Management

### List Documents: `GET /api/v1/documents/`

```python
@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(20, description="Items per page", ge=1, le=100),
    search: Optional[str] = Query(None, description="Search query for document titles"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    document_service: DocumentService = Depends(get_document_service)
):
```

#### Pagination and Filtering

```python
# Calculate offset for pagination
offset = (page - 1) * page_size

# Build query with filters
query = self.supabase.table("documents").select("*", count="exact")

if user_id:
    query = query.eq("user_id", user_id)

if search_query:
    query = query.ilike("title", f"%{search_query}%")

# Apply pagination and ordering
query = query.order("created_at", desc=True).range(offset, offset + page_size - 1)

result = query.execute()
```

### Get Specific Document: `GET /api/v1/documents/{document_id}`

```python
@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user_id: Optional[str] = Query(None, description="User ID for ownership verification"),
    document_service: DocumentService = Depends(get_document_service)
):
```

## 🔍 Search Operations

### Semantic Search: `POST /api/v1/documents/search`

```python
@router.post("/search", response_model=SearchResponse)
async def search_documents(
    search_request: SearchRequest,
    user_id: Optional[str] = Query(None, description="User ID for filtering results"),
    openai_service: OpenAIService = Depends(get_openai_service),
    document_service: DocumentService = Depends(get_document_service)
):
```

#### Search Request Flow

```python
# Step 1: Generate query embedding
start_time = time.time()
query_embeddings = await openai_service.create_embeddings_batch([search_request.query])
query_embedding = query_embeddings[0]

# Step 2: Perform vector search
search_results = await document_service.search_documents(
    query_embedding=query_embedding,
    limit=search_request.limit,
    threshold=search_request.threshold,
    document_ids=search_request.document_ids,
    user_id=user_id
)

# Step 3: Calculate search time and return results
search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
```

#### Search Request Model

```python
{
    "query": "machine learning algorithms for text processing",
    "max_results": 10,
    "similarity_threshold": 0.7,
    "document_ids": ["doc1_id", "doc2_id"],  # Optional filter
    "user_id": "user_123",                   # Optional filter
    "boost_recent": true,
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "metadata_filters": {
        "category": "research",
        "author": "John Doe"
    }
}
```

#### Search Response Model

```python
{
    "query": "machine learning algorithms for text processing",
    "search_type": "semantic",
    "results": [
        {
            "chunk_id": "chunk_123",
            "document_id": "doc_456",
            "filename": "ml_research.pdf",
            "chunk_text": "Machine learning algorithms have revolutionized text processing...",
            "similarity": 0.89,
            "chunk_index": 5,
            "metadata": {
                "page_number": 3,
                "section": "Introduction"
            }
        }
    ],
    "total_results": 25,
    "filtered_results": 10,
    "query_time": 0.234,
    "avg_similarity": 0.82,
    "metadata": {
        "embedding_model": "text-embedding-ada-002",
        "search_strategy": "cosine_similarity"
    }
}
```

## 🔄 Document Management Operations

### Update Document Metadata: `PUT /api/v1/documents/{document_id}/metadata`

```python
@router.put("/{document_id}/metadata", response_model=DocumentResponse)
async def update_document_metadata(
    document_id: str,
    update_request: DocumentUpdateRequest,
    user_id: Optional[str] = Query(None),
    document_service: DocumentService = Depends(get_document_service)
):
```

### Delete Document: `DELETE /api/v1/documents/{document_id}`

```python
@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    user_id: Optional[str] = Query(None),
    document_service: DocumentService = Depends(get_document_service)
):
```

#### Deletion Process

```python
# Delete embeddings first (foreign key constraint)
embedding_delete = self.supabase.table("embeddings").delete().eq("document_id", document_id)

if user_id:
    # Verify ownership before deletion
    document_query = self.supabase.table("documents").select("user_id").eq("id", document_id)
    if user_id:
        document_query = document_query.eq("user_id", user_id)

    document_result = document_query.execute()
    if not document_result.data:
        return False

# Delete document record
document_delete = self.supabase.table("documents").delete().eq("id", document_id)
if user_id:
    document_delete = document_delete.eq("user_id", user_id)

result = document_delete.execute()
return len(result.data) > 0
```

## 📈 Analytics and Statistics

### Document Statistics: `GET /api/v1/documents/stats/overview`

```python
@router.get("/stats/overview", response_model=DocumentStats)
async def get_document_stats(
    user_id: Optional[str] = Query(None),
    document_service: DocumentService = Depends(get_document_service)
):
```

#### Statistics Calculation

```python
# Document count
doc_query = self.supabase.table("documents").select("id", count="exact")
if user_id:
    doc_query = doc_query.eq("user_id", user_id)
doc_result = doc_query.execute()
total_documents = doc_result.count or 0

# Chunk count
chunk_query = self.supabase.table("embeddings").select("id", count="exact")
chunk_result = chunk_query.execute()
total_chunks = chunk_result.count or 0

# Token count (calculated from chunks)
token_query = self.supabase.table("embeddings").select("chunk_metadata")
token_result = token_query.execute()
total_tokens = sum(
    chunk.get("chunk_metadata", {}).get("token_count", 0)
    for chunk in token_result.data
)
```

## ⚠️ Error Handling

### Error Response Format

```python
# Standardized error response
{
    "error": "Document processing failed",
    "error_code": "PROCESSING_ERROR",
    "details": {
        "stage": "text_extraction",
        "filename": "document.pdf",
        "reason": "PDF file is password protected",
        "suggestion": "Please provide an unprotected PDF file"
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Scenarios

1. **File Upload Errors**

   - File size exceeds limits
   - Unsupported file format
   - Corrupted file content
   - Missing filename

2. **Processing Errors**

   - Text extraction failures
   - Chunking errors
   - Embedding generation failures
   - Database storage errors

3. **Access Control Errors**

   - Document not found
   - Access denied for user
   - Invalid document ID

4. **Validation Errors**
   - Invalid chunk size parameters
   - Malformed metadata JSON
   - Invalid search parameters

## 📊 Performance Monitoring

### Request Metrics

```python
# Tracked metrics per request
request_metrics = {
    "upload_time": "Time from upload to storage completion",
    "processing_time": "Document processing duration",
    "embedding_time": "Embedding generation duration",
    "storage_time": "Database storage duration",
    "file_size": "Original file size",
    "chunk_count": "Number of chunks generated",
    "token_count": "Total tokens processed"
}
```

### Batch Processing Metrics

```python
# Batch operation tracking
batch_metrics = {
    "total_files": "Number of files in batch",
    "successful_uploads": "Successfully processed files",
    "failed_uploads": "Failed processing attempts",
    "total_processing_time": "Complete batch processing time",
    "average_file_processing_time": "Average time per file",
    "throughput": "Files processed per second"
}
```

This comprehensive API request flow documentation provides developers with detailed understanding of how document processing requests move through the system, enabling effective integration and troubleshooting.

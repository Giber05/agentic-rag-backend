# Document Processing Sequence Diagrams

## 🎯 Overview

This document provides detailed sequence diagrams showing the complete document processing flow from file upload to storage and retrieval. It covers single uploads, batch processing, search operations, and all service interactions.

## 🔄 Complete Document Upload Processing Sequence

The following diagram shows the complete flow of a document upload through the processing pipeline:

```mermaid
sequenceDiagram
    participant Client
    participant API as "FastAPI Documents Endpoint"
    participant DocProcessor as "Document Processor"
    participant DocService as "Document Service"
    participant OpenAI as "OpenAI Service"
    participant Supabase as "Supabase Database"

    Client->>API: POST /api/v1/documents/upload
    Note over Client,API: Multipart form data with file, chunk_size, metadata

    API->>API: Dependency Injection
    API->>API: Validate file upload parameters

    alt File validation fails
        API-->>Client: HTTP 400 - Validation Error
    else File validation passes
        API->>DocProcessor: validate_file(file_content, filename)
        DocProcessor->>DocProcessor: detect_content_type()
        DocProcessor->>DocProcessor: check_file_size()
        DocProcessor-->>API: validation_result

        alt File invalid
            API-->>Client: HTTP 400 - File Invalid
        else File valid
            API->>DocProcessor: process_document()

            Note over DocProcessor: Stage 1: Text Extraction
            DocProcessor->>DocProcessor: _extract_text(content_type)

            alt PDF file
                DocProcessor->>DocProcessor: _extract_text_pdf()
            else DOCX file
                DocProcessor->>DocProcessor: _extract_text_docx()
            else Plain text
                DocProcessor->>DocProcessor: _extract_text_plain()
            else HTML file
                DocProcessor->>DocProcessor: _extract_text_html()
            end

            Note over DocProcessor: Stage 2: Text Chunking
            DocProcessor->>DocProcessor: _create_chunks()
            DocProcessor->>DocProcessor: sentence_tokenize()
            DocProcessor->>DocProcessor: calculate_token_counts()
            DocProcessor->>DocProcessor: apply_overlap_strategy()

            Note over DocProcessor: Stage 3: Embedding Generation
            DocProcessor->>OpenAI: create_embeddings_batch(chunk_texts)
            OpenAI-->>DocProcessor: embeddings[]
            DocProcessor->>DocProcessor: assign_embeddings_to_chunks()

            DocProcessor-->>API: ProcessingResult

            Note over API: Stage 4: Database Storage
            API->>DocService: store_document(processing_result)
            DocService->>DocService: generate_document_id()
            DocService->>Supabase: INSERT INTO documents
            Supabase-->>DocService: document_stored

            DocService->>DocService: _store_embeddings()
            DocService->>Supabase: INSERT INTO embeddings (batch)
            Supabase-->>DocService: embeddings_stored

            DocService-->>API: DocumentResponse
            API-->>Client: HTTP 201 - Success with DocumentResponse
        end
    end
```

## 🔄 Batch Upload Processing Sequence

This diagram shows how multiple documents are processed in a batch operation:

```mermaid
sequenceDiagram
    participant Client
    participant API as "Batch Upload Endpoint"
    participant DocProcessor as "Document Processor"
    participant DocService as "Document Service"
    participant OpenAI as "OpenAI Service"

    Client->>API: POST /api/v1/documents/batch-upload
    Note over Client,API: Multiple files + batch parameters

    API->>API: Initialize batch tracking
    API->>API: start_time = time.time()
    API->>API: successful_uploads = []
    API->>API: failed_uploads = []

    loop For each file in batch
        API->>API: read file content

        alt File validation fails
            API->>API: add to failed_uploads
        else File validation passes
            API->>DocProcessor: process_document(file)

            Note over DocProcessor: Individual Processing Pipeline
            DocProcessor->>DocProcessor: extract_text()
            DocProcessor->>DocProcessor: create_chunks()
            DocProcessor->>OpenAI: generate_embeddings()
            OpenAI-->>DocProcessor: embeddings
            DocProcessor-->>API: ProcessingResult

            API->>DocService: store_document()
            DocService-->>API: DocumentResponse
            API->>API: add to successful_uploads
        end
    end

    API->>API: calculate_processing_time()
    API->>API: compile_batch_results()
    API-->>Client: BatchUploadResponse
    Note over API,Client: Success/failure counts + processing time
```

## 🔍 Document Search Sequence

This diagram shows the semantic search process:

```mermaid
sequenceDiagram
    participant Client
    participant API as "Search Endpoint"
    participant OpenAI as "OpenAI Service"
    participant DocService as "Document Service"
    participant Supabase as "Supabase Vector DB"

    Client->>API: POST /api/v1/documents/search
    Note over Client,API: SearchRequest with query, filters, parameters

    API->>API: start_time = time.time()
    API->>API: validate search parameters

    Note over API: Stage 1: Query Embedding
    API->>OpenAI: create_embeddings_batch([query])
    OpenAI-->>API: query_embedding[1536]

    Note over API: Stage 2: Vector Search
    API->>DocService: search_documents()
    DocService->>DocService: build_search_query()
    DocService->>Supabase: RPC search_embeddings_with_metadata()
    Note over DocService,Supabase: Vector similarity search with filters
    Supabase-->>DocService: ranked_results[]

    DocService->>DocService: post_process_results()
    DocService->>DocService: apply_recency_boost()
    DocService->>DocService: filter_by_threshold()
    DocService->>DocService: apply_metadata_filters()
    DocService-->>API: SearchResult[]

    API->>API: calculate_search_time()
    API->>API: compile_search_response()
    API-->>Client: SearchResponse
    Note over API,Client: Results + metadata + performance stats
```

## 🗂️ Document Management Operations

This diagram shows CRUD operations for documents:

```mermaid
sequenceDiagram
    participant Client
    participant API as "Management Endpoints"
    participant DocService as "Document Service"
    participant Supabase as "Supabase Database"

    Note over Client,API: Document Listing
    Client->>API: GET /api/v1/documents/?page=1&page_size=20
    API->>DocService: list_documents(page, page_size, filters)
    DocService->>DocService: calculate_offset()
    DocService->>Supabase: SELECT with pagination
    Supabase-->>DocService: documents_page
    DocService->>Supabase: COUNT chunks per document
    Supabase-->>DocService: chunk_counts
    DocService-->>API: DocumentListResponse
    API-->>Client: Paginated document list

    Note over Client,API: Document Retrieval
    Client->>API: GET /api/v1/documents/{document_id}
    API->>DocService: get_document(document_id, user_id)
    DocService->>Supabase: SELECT document by ID
    alt Document not found
        DocService-->>API: None
        API-->>Client: HTTP 404 - Not Found
    else Document found
        Supabase-->>DocService: document_data
        DocService->>Supabase: COUNT chunks
        Supabase-->>DocService: chunk_count
        DocService-->>API: DocumentResponse
        API-->>Client: Document details
    end

    Note over Client,API: Document Deletion
    Client->>API: DELETE /api/v1/documents/{document_id}
    API->>DocService: delete_document(document_id, user_id)
    DocService->>DocService: verify_ownership()
    DocService->>Supabase: DELETE FROM embeddings
    Supabase-->>DocService: embeddings_deleted
    DocService->>Supabase: DELETE FROM documents
    Supabase-->>DocService: document_deleted
    DocService-->>API: deletion_success
    API-->>Client: HTTP 204 - No Content
```

## 🏗️ Service Integration Overview

This diagram shows how all services integrate in the document processing system:

```mermaid
graph TB
    subgraph "API Layer"
        Upload[Upload Endpoint]
        Batch[Batch Upload Endpoint]
        Search[Search Endpoint]
        Management[Management Endpoints]
    end

    subgraph "Processing Layer"
        DocProcessor[Document Processor]
        DocService[Document Service]
    end

    subgraph "Service Layer"
        OpenAIService[OpenAI Service]
        Cache[Cache Service]
        RateLimit[Rate Limiter]
    end

    subgraph "Data Layer"
        Supabase[(Supabase Database)]
        OpenAI_API[OpenAI API]
    end

    Upload --> DocProcessor
    Batch --> DocProcessor
    Search --> DocService
    Management --> DocService

    DocProcessor --> OpenAIService
    DocProcessor --> DocService
    DocService --> Supabase

    OpenAIService --> Cache
    OpenAIService --> RateLimit
    OpenAIService --> OpenAI_API

    DocService --> Supabase
```

## 🚨 Error Handling Sequence

This diagram shows error handling throughout the processing pipeline:

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant DocProcessor
    participant OpenAI
    participant Supabase
    participant ErrorHandler as "Error Handler"

    Client->>API: Document Upload Request

    alt File Validation Error
        API->>API: validate_file()
        API->>ErrorHandler: handle_validation_error()
        ErrorHandler-->>API: formatted_error_response
        API-->>Client: HTTP 400 - Validation Error
    else Text Extraction Error
        API->>DocProcessor: process_document()
        DocProcessor->>DocProcessor: extract_text()
        DocProcessor->>ErrorHandler: handle_extraction_error()
        ErrorHandler-->>DocProcessor: fallback_strategy
        DocProcessor-->>API: partial_processing_result
        API-->>Client: HTTP 206 - Partial Success
    else OpenAI API Error
        DocProcessor->>OpenAI: generate_embeddings()
        OpenAI-->>DocProcessor: API Error (rate limit/timeout)
        DocProcessor->>ErrorHandler: handle_openai_error()

        alt Retry Success
            ErrorHandler->>OpenAI: retry with backoff
            OpenAI-->>ErrorHandler: success
            ErrorHandler-->>DocProcessor: embeddings
        else Retry Failed
            ErrorHandler-->>DocProcessor: embedding_generation_failed
            DocProcessor-->>API: processing_failed
            API-->>Client: HTTP 503 - Service Unavailable
        end
    else Database Error
        API->>Supabase: store_document()
        Supabase-->>API: Database Error
        API->>ErrorHandler: handle_database_error()
        ErrorHandler-->>API: formatted_error_response
        API-->>Client: HTTP 500 - Internal Server Error
    else Success Path
        API->>DocProcessor: process_document()
        DocProcessor->>OpenAI: generate_embeddings()
        OpenAI-->>DocProcessor: embeddings
        DocProcessor-->>API: processing_result
        API->>Supabase: store_document()
        Supabase-->>API: stored_document
        API-->>Client: HTTP 201 - Success
    end
```

## 📊 Performance Monitoring Flow

This diagram shows how performance metrics are collected throughout the processing pipeline:

```mermaid
sequenceDiagram
    participant Request
    participant Metrics as "Performance Metrics"
    participant DocProcessor
    participant OpenAI
    participant Supabase
    participant Analytics as "Analytics Store"

    Request->>Metrics: start_request_tracking()
    Metrics->>Metrics: record_start_time()

    Request->>DocProcessor: process_document()
    DocProcessor->>Metrics: track_stage("text_extraction")
    DocProcessor->>DocProcessor: extract_text()
    DocProcessor->>Metrics: complete_stage("text_extraction", duration)

    DocProcessor->>Metrics: track_stage("chunking")
    DocProcessor->>DocProcessor: create_chunks()
    DocProcessor->>Metrics: complete_stage("chunking", duration)

    DocProcessor->>Metrics: track_stage("embedding_generation")
    DocProcessor->>OpenAI: generate_embeddings()
    OpenAI-->>DocProcessor: embeddings
    DocProcessor->>Metrics: complete_stage("embedding_generation", duration)

    Request->>Metrics: track_stage("database_storage")
    Request->>Supabase: store_document()
    Supabase-->>Request: stored
    Request->>Metrics: complete_stage("database_storage", duration)

    Request->>Metrics: finish_request_tracking()
    Metrics->>Metrics: calculate_total_time()
    Metrics->>Metrics: calculate_throughput()
    Metrics->>Analytics: store_performance_data()

    Note over Metrics,Analytics: Cost tracking, processing times, throughput
```

## 🔄 Dependency Injection Flow

This diagram shows how dependencies are injected and managed:

```mermaid
sequenceDiagram
    participant FastAPI
    participant Dependencies as "dependencies.py"
    participant DocProcessor
    participant DocService
    participant OpenAI
    participant Supabase

    FastAPI->>Dependencies: get_document_processor()

    alt First call (not cached)
        Dependencies->>Dependencies: get_openai_service()
        Dependencies->>OpenAI: create instance
        Dependencies->>DocProcessor: create instance with OpenAI service
        Dependencies-->>FastAPI: processor instance
    else Subsequent calls (cached)
        Dependencies-->>FastAPI: cached processor
    end

    FastAPI->>Dependencies: get_document_service()

    alt First call (not cached)
        Dependencies->>Dependencies: get_supabase_client()
        Dependencies->>Supabase: create_client()
        Dependencies->>DocService: create instance with Supabase client
        Dependencies-->>FastAPI: service instance
    else Subsequent calls (cached)
        Dependencies-->>FastAPI: cached service
    end
```

## 📝 Key Insights from Sequence Diagrams

### Processing Optimization

1. **Batch Processing**: Efficient handling of multiple files with error isolation
2. **Streaming Embeddings**: Batch embedding generation reduces API calls
3. **Database Optimization**: Bulk insert operations for embeddings
4. **Error Recovery**: Graceful degradation with partial processing

### Performance Characteristics

1. **Text Extraction**: Format-specific optimizations reduce processing time
2. **Chunking Strategy**: Sentence-based boundaries improve semantic coherence
3. **Embedding Efficiency**: Batch operations maximize API throughput
4. **Storage Optimization**: Structured data storage with proper indexing

### Error Resilience

1. **Validation Gates**: Multiple validation layers prevent processing failures
2. **Retry Logic**: Exponential backoff for external API failures
3. **Partial Success**: Continue processing when individual files fail
4. **Resource Cleanup**: Proper cleanup on processing failures

### Scalability Features

1. **Async Processing**: Non-blocking operations throughout pipeline
2. **Connection Pooling**: Efficient database connection management
3. **Rate Limiting**: API quota management for external services
4. **Batch Operations**: Reduced overhead for bulk processing

These sequence diagrams provide a comprehensive view of the document processing system's architecture and data flow, enabling developers to understand the complete processing lifecycle and optimization strategies.

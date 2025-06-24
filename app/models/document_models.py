from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

class DocumentChunk(BaseModel):
    """Model for a document chunk with embedding"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(..., description="The text content of the chunk")
    chunk_index: int = Field(..., description="Index of this chunk within the document")
    token_count: int = Field(..., description="Number of tokens in this chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for this chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the chunk")
    
    class Config:
        json_encoders = {
            UUID: str
        }

class ProcessingResult(BaseModel):
    """Result of document processing"""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the document")
    text_content: str = Field(..., description="Extracted text content")
    chunks: List[DocumentChunk] = Field(..., description="Generated chunks with embeddings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    
    class Config:
        json_encoders = {
            UUID: str
        }

class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    filename: str = Field(..., description="Name of the file")
    content_type: Optional[str] = Field(None, description="MIME type (auto-detected if not provided)")
    chunk_size: int = Field(1000, description="Target chunk size in tokens", ge=100, le=4000)
    chunk_overlap: int = Field(200, description="Overlap between chunks in tokens", ge=0, le=1000)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DocumentResponse(BaseModel):
    """Response model for document operations"""
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., description="File size in bytes")
    chunk_count: int = Field(..., description="Number of chunks generated")
    created_at: datetime = Field(..., description="Upload timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")

class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    documents: List[DocumentResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(20, description="Number of items per page")

class ChunkResponse(BaseModel):
    """Response model for document chunks"""
    id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Parent document ID")
    text: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index within document")
    token_count: int = Field(..., description="Number of tokens")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

class SearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Search query", min_length=1)
    max_results: int = Field(10, description="Maximum number of results", ge=1, le=100)
    similarity_threshold: float = Field(0.7, description="Similarity threshold", ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific document IDs")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    boost_recent: bool = Field(True, description="Boost recent documents in ranking")
    semantic_weight: Optional[float] = Field(0.7, description="Weight for semantic search in hybrid mode", ge=0.0, le=1.0)
    keyword_weight: Optional[float] = Field(0.3, description="Weight for keyword search in hybrid mode", ge=0.0, le=1.0)
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Filter by metadata")

class SearchResult(BaseModel):
    """Individual search result"""
    chunk_id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Source filename")
    chunk_text: str = Field(..., description="Chunk text content")
    similarity: float = Field(..., description="Similarity score")
    chunk_index: int = Field(..., description="Chunk index in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str = Field(..., description="Original search query")
    search_type: str = Field(..., description="Type of search performed (semantic, keyword, hybrid)")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of matching chunks before filtering")
    filtered_results: int = Field(..., description="Number of results after filtering")
    query_time: float = Field(..., description="Total query execution time in seconds")
    avg_similarity: float = Field(..., description="Average similarity score of results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional search metadata")
    
class DocumentStats(BaseModel):
    """Document processing statistics"""
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_tokens: int = Field(..., description="Total number of tokens processed")
    supported_formats: List[str] = Field(..., description="List of supported file formats")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Additional processing statistics")

class BatchUploadRequest(BaseModel):
    """Request model for batch document upload"""
    files: List[DocumentUploadRequest] = Field(..., description="List of files to upload")
    default_chunk_size: int = Field(1000, description="Default chunk size for all files", ge=100, le=4000)
    default_chunk_overlap: int = Field(200, description="Default overlap for all files", ge=0, le=1000)
    
class BatchUploadResponse(BaseModel):
    """Response model for batch upload"""
    successful_uploads: List[DocumentResponse] = Field(..., description="Successfully processed documents")
    failed_uploads: List[Dict[str, str]] = Field(..., description="Failed uploads with error messages")
    total_processed: int = Field(..., description="Total number of files processed")
    success_count: int = Field(..., description="Number of successful uploads")
    failure_count: int = Field(..., description="Number of failed uploads")
    processing_time_seconds: float = Field(..., description="Total processing time")

class DocumentUpdateRequest(BaseModel):
    """Request model for updating document metadata"""
    metadata: Dict[str, Any] = Field(..., description="Updated metadata")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp") 
import logging
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from ...core.dependencies import get_supabase_client, get_openai_service, get_document_processor, get_document_service
from ...models.document_models import (
    DocumentResponse,
    DocumentListResponse,
    SearchRequest,
    SearchResponse,
    DocumentStats,
    DocumentUpdateRequest,
    ErrorResponse,
    BatchUploadResponse
)
from ...services.document_processor import DocumentProcessor
from ...services.document_service import DocumentService
from ...services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000, description="Target chunk size in tokens", ge=100, le=4000),
    chunk_overlap: int = Form(200, description="Overlap between chunks in tokens", ge=0, le=1000),
    metadata: Optional[str] = Form(None, description="Additional metadata as JSON string"),
    user_id: Optional[str] = Form(None, description="User ID for ownership"),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Upload and process a document
    
    Supports multiple file formats:
    - PDF (.pdf)
    - Word documents (.docx)
    - Plain text (.txt)
    - HTML (.html)
    - Markdown (.md)
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Validate file
        is_valid, validation_message = await document_processor.validate_file(
            file_content, file.filename
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation_message
            )
        
        # Parse metadata if provided
        additional_metadata = {}
        if metadata:
            try:
                import json
                additional_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid metadata JSON format"
                )
        
        # Process document
        logger.info(f"Processing document: {file.filename}")
        processing_result = await document_processor.process_document(
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=additional_metadata
        )
        
        # Store in database
        document_response = await document_service.store_document(
            processing_result, user_id
        )
        
        logger.info(f"Successfully uploaded and processed document: {file.filename}")
        return document_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )

@router.post("/batch-upload", response_model=BatchUploadResponse)
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(1000, description="Default chunk size for all files"),
    chunk_overlap: int = Form(200, description="Default overlap for all files"),
    user_id: Optional[str] = Form(None, description="User ID for ownership"),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    document_service: DocumentService = Depends(get_document_service)
):
    """Upload and process multiple documents in batch"""
    try:
        start_time = time.time()
        successful_uploads = []
        failed_uploads = []
        
        for file in files:
            try:
                if not file.filename:
                    failed_uploads.append({
                        "filename": "unknown",
                        "error": "Filename is required"
                    })
                    continue
                
                # Read and validate file
                file_content = await file.read()
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
        
        return BatchUploadResponse(
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            total_processed=len(files),
            success_count=len(successful_uploads),
            failure_count=len(failed_uploads),
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upload failed: {str(e)}"
        )

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(20, description="Items per page", ge=1, le=100),
    search: Optional[str] = Query(None, description="Search query for document titles"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    document_service: DocumentService = Depends(get_document_service)
):
    """List documents with pagination and optional search"""
    try:
        documents, total_count = await document_service.list_documents(
            user_id=user_id,
            page=page,
            page_size=page_size,
            search_query=search
        )
        
        return DocumentListResponse(
            documents=documents,
            total=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user_id: Optional[str] = Query(None, description="User ID for ownership verification"),
    document_service: DocumentService = Depends(get_document_service)
):
    """Get a specific document by ID"""
    try:
        document = await document_service.get_document(document_id, user_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )

@router.put("/{document_id}/metadata", response_model=DocumentResponse)
async def update_document_metadata(
    document_id: str,
    update_request: DocumentUpdateRequest,
    user_id: Optional[str] = Query(None, description="User ID for ownership verification"),
    document_service: DocumentService = Depends(get_document_service)
):
    """Update document metadata"""
    try:
        success = await document_service.update_document_metadata(
            document_id, update_request.metadata, user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or access denied"
            )
        
        # Return updated document
        updated_document = await document_service.get_document(document_id, user_id)
        return updated_document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document metadata {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document metadata: {str(e)}"
        )

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    user_id: Optional[str] = Query(None, description="User ID for ownership verification"),
    document_service: DocumentService = Depends(get_document_service)
):
    """Delete a document and all its chunks"""
    try:
        success = await document_service.delete_document(document_id, user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or access denied"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    search_request: SearchRequest,
    user_id: Optional[str] = Query(None, description="User ID for filtering results"),
    openai_service: OpenAIService = Depends(get_openai_service),
    document_service: DocumentService = Depends(get_document_service)
):
    """Perform semantic search across documents"""
    try:
        start_time = time.time()
        
        # Generate embedding for search query
        query_embeddings = await openai_service.create_embeddings_batch([search_request.query])
        query_embedding = query_embeddings[0]
        
        # Perform vector search
        search_results = await document_service.search_documents(
            query_embedding=query_embedding,
            limit=search_request.limit,
            threshold=search_request.threshold,
            document_ids=search_request.document_ids,
            user_id=user_id
        )
        
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return SearchResponse(
            query=search_request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Error performing document search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    user_id: Optional[str] = Query(None, description="User ID for ownership verification"),
    document_service: DocumentService = Depends(get_document_service)
):
    """Get all chunks for a specific document"""
    try:
        chunks = await document_service.get_document_chunks(document_id, user_id)
        
        if not chunks:
            # Check if document exists
            document = await document_service.get_document(document_id, user_id)
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )
        
        return {"chunks": chunks}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunks for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document chunks: {str(e)}"
        )

@router.get("/stats/overview", response_model=DocumentStats)
async def get_document_stats(
    user_id: Optional[str] = Query(None, description="User ID for filtering stats"),
    document_service: DocumentService = Depends(get_document_service)
):
    """Get document processing statistics"""
    try:
        stats = await document_service.get_document_stats(user_id)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document stats: {str(e)}"
        )

@router.get("/formats/supported")
async def get_supported_formats(
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """Get list of supported file formats"""
    try:
        formats = document_processor.get_supported_formats()
        return {
            "supported_formats": formats,
            "format_descriptions": {
                "text/plain": "Plain text files (.txt)",
                "application/pdf": "PDF documents (.pdf)",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word documents (.docx)",
                "text/html": "HTML files (.html)",
                "text/markdown": "Markdown files (.md)",
                "text/csv": "CSV files (.csv)"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting supported formats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get supported formats: {str(e)}"
        ) 
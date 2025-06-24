"""
Simplified document endpoints for testing without external dependencies
"""

from fastapi import APIRouter
from typing import List, Dict, Any

router = APIRouter(prefix="/documents-simple", tags=["documents-simple"])

@router.get("/formats/supported")
async def get_supported_formats() -> Dict[str, Any]:
    """Get list of supported document formats."""
    return {
        "supported_formats": [
            "text/plain",
            "application/pdf", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/html",
            "application/json"
        ],
        "max_file_size_mb": 50,
        "description": "Supported document formats for processing"
    }

@router.get("/stats/overview")
async def get_stats_overview() -> Dict[str, Any]:
    """Get document processing statistics overview."""
    return {
        "total_documents": 0,
        "total_chunks": 0,
        "total_embeddings": 0,
        "storage_used_mb": 0.0,
        "last_processed": None,
        "processing_status": "ready"
    }

@router.get("/health")
async def document_service_health() -> Dict[str, str]:
    """Check document service health."""
    return {
        "status": "healthy",
        "service": "document_processor",
        "version": "1.0.0"
    } 
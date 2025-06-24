"""
Database API endpoints for testing and management.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from uuid import UUID

from ..database.connection import test_database_connection, check_vector_extension, execute_migration
from ..services.database import db_service
from ..models.database import (
    Document, DocumentCreate, DocumentUpdate,
    Conversation, ConversationCreate,
    Message, MessageCreate,
    VectorSearchRequest, VectorSearchResponse
)
from ..models.base import BaseAPIModel
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class DatabaseStatus(BaseAPIModel):
    """Database status response model."""
    connected: bool
    pgvector_available: bool
    message: str


class MigrationRequest(BaseAPIModel):
    """Migration request model."""
    migration_name: str
    sql_content: str


class MigrationResponse(BaseAPIModel):
    """Migration response model."""
    success: bool
    message: str


@router.get("/database/status", response_model=DatabaseStatus)
async def get_database_status():
    """
    Check database connection and pgvector availability.
    
    Returns:
        DatabaseStatus: Database connectivity status
    """
    try:
        # Test basic connection
        connected = await test_database_connection()
        
        # Check pgvector extension
        pgvector_available = False
        if connected:
            pgvector_available = await check_vector_extension()
        
        if connected and pgvector_available:
            message = "Database connected and pgvector extension available"
        elif connected:
            message = "Database connected but pgvector extension not available"
        else:
            message = "Database connection failed"
        
        logger.info("Database status checked", connected=connected, pgvector=pgvector_available)
        
        return DatabaseStatus(
            connected=connected,
            pgvector_available=pgvector_available,
            message=message
        )
        
    except ValueError as e:
        # Configuration errors
        logger.error("Database configuration error", error=str(e))
        return DatabaseStatus(
            connected=False,
            pgvector_available=False,
            message=f"Database not configured: {str(e)}"
        )
    except Exception as e:
        logger.error("Database status check failed", error=str(e))
        if "nodename nor servname provided" in str(e):
            return DatabaseStatus(
                connected=False,
                pgvector_available=False,
                message="Database connection failed: Unable to resolve database hostname. The Supabase project may be paused or the database URL is incorrect."
            )
        return DatabaseStatus(
            connected=False,
            pgvector_available=False,
            message=f"Database status check failed: {str(e)}"
        )


@router.post("/database/migrate", response_model=MigrationResponse)
async def run_migration(migration: MigrationRequest):
    """
    Execute a database migration.
    
    Args:
        migration: Migration request with SQL content
        
    Returns:
        MigrationResponse: Migration execution result
    """
    try:
        success = await execute_migration(migration.sql_content)
        
        if success:
            message = f"Migration '{migration.migration_name}' executed successfully"
            logger.info("Migration executed", migration_name=migration.migration_name)
        else:
            message = f"Migration '{migration.migration_name}' failed"
            logger.error("Migration failed", migration_name=migration.migration_name)
        
        return MigrationResponse(success=success, message=message)
        
    except Exception as e:
        logger.error("Migration execution failed", migration_name=migration.migration_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


# Document endpoints for testing
@router.post("/database/documents", response_model=Document)
async def create_document_test(document: DocumentCreate):
    """Create a test document."""
    try:
        await db_service.initialize()
        result = await db_service.create_document(document)
        return result
    except Exception as e:
        logger.error("Document creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Document creation failed: {str(e)}")


@router.get("/database/documents", response_model=List[Document])
async def list_documents_test(limit: int = 10, offset: int = 0):
    """List test documents."""
    try:
        await db_service.initialize()
        result = await db_service.list_documents(limit=limit, offset=offset)
        return result
    except Exception as e:
        logger.error("Document listing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Document listing failed: {str(e)}")


@router.get("/database/documents/{document_id}", response_model=Document)
async def get_document_test(document_id: UUID):
    """Get a test document by ID."""
    try:
        await db_service.initialize()
        result = await db_service.get_document(document_id)
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document retrieval failed", document_id=str(document_id), error=str(e))
        raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")


# Conversation endpoints for testing
@router.post("/database/conversations", response_model=Conversation)
async def create_conversation_test(conversation: ConversationCreate):
    """Create a test conversation."""
    try:
        await db_service.initialize()
        result = await db_service.create_conversation(conversation)
        return result
    except Exception as e:
        logger.error("Conversation creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Conversation creation failed: {str(e)}")


@router.get("/database/conversations", response_model=List[Conversation])
async def list_conversations_test(limit: int = 10, offset: int = 0):
    """List test conversations."""
    try:
        await db_service.initialize()
        result = await db_service.list_conversations(limit=limit, offset=offset)
        return result
    except ValueError as e:
        # Configuration errors (missing Supabase URL, etc.)
        logger.error("Database configuration error", error=str(e))
        raise HTTPException(status_code=503, detail=f"Database not configured: {str(e)}")
    except Exception as e:
        # Connection errors, DNS issues, etc.
        logger.error("Conversation listing failed", error=str(e))
        if "nodename nor servname provided" in str(e):
            raise HTTPException(
                status_code=503, 
                detail="Database connection failed: Unable to resolve database hostname. The Supabase project may be paused or the database URL is incorrect."
            )
        raise HTTPException(status_code=500, detail=f"Conversation listing failed: {str(e)}")


@router.post("/database/conversations/{conversation_id}/messages", response_model=Message)
async def create_message_test(conversation_id: UUID, message: MessageCreate):
    """Create a test message in a conversation."""
    try:
        await db_service.initialize()
        
        # Ensure the conversation exists
        conversation = await db_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Set the conversation_id from the URL
        message.conversation_id = conversation_id
        result = await db_service.create_message(message)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Message creation failed", conversation_id=str(conversation_id), error=str(e))
        raise HTTPException(status_code=500, detail=f"Message creation failed: {str(e)}")


@router.get("/database/conversations/{conversation_id}/messages", response_model=List[Message])
async def get_conversation_messages_test(conversation_id: UUID, limit: int = 100):
    """Get messages for a test conversation."""
    try:
        await db_service.initialize()
        result = await db_service.get_conversation_messages(conversation_id, limit=limit)
        return result
    except Exception as e:
        logger.error("Message retrieval failed", conversation_id=str(conversation_id), error=str(e))
        raise HTTPException(status_code=500, detail=f"Message retrieval failed: {str(e)}") 
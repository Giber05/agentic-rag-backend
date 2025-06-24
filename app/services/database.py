"""
Database service for managing CRUD operations and vector search.
"""

import time
from typing import List, Optional, Dict, Any
from uuid import UUID
import asyncpg
from ..database.connection import get_asyncpg_pool, get_supabase_client
from ..models.database import (
    Document, DocumentCreate, DocumentUpdate,
    Embedding, EmbeddingCreate,
    Conversation, ConversationCreate, ConversationUpdate,
    Message, MessageCreate,
    AgentLog, AgentLogCreate,
    VectorSearchRequest, VectorSearchResult, VectorSearchResponse
)
from ..core.logging import get_logger

logger = get_logger(__name__)


class DatabaseService:
    """Service for database operations."""
    
    def __init__(self):
        self.supabase = None
        self.pool = None
    
    async def initialize(self):
        """Initialize database connections."""
        try:
            self.supabase = get_supabase_client()
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Supabase client", error=str(e))
            raise
        
        try:
            self.pool = await get_asyncpg_pool()
            logger.info("AsyncPG pool initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize AsyncPG pool", error=str(e))
            raise
    
    # Document operations
    async def create_document(self, document: DocumentCreate) -> Document:
        """Create a new document."""
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO documents (title, content, file_type, file_size, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                document.title,
                document.content,
                document.file_type,
                document.file_size,
                document.metadata
            )
            
        logger.info("Document created", document_id=str(row['id']))
        return Document(**dict(row))
    
    async def get_document(self, document_id: UUID) -> Optional[Document]:
        """Get a document by ID."""
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM documents WHERE id = $1",
                document_id
            )
            
        if row:
            return Document(**dict(row))
        return None
    
    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """List documents with pagination."""
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM documents 
                ORDER BY created_at DESC 
                LIMIT $1 OFFSET $2
                """,
                limit, offset
            )
            
        return [Document(**dict(row)) for row in rows]
    
    async def update_document(self, document_id: UUID, update: DocumentUpdate) -> Optional[Document]:
        """Update a document."""
        update_data = update.dict(exclude_unset=True)
        if not update_data:
            return await self.get_document(document_id)
        
        set_clause = ", ".join([f"{key} = ${i+2}" for i, key in enumerate(update_data.keys())])
        values = [document_id] + list(update_data.values())
        
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                f"""
                UPDATE documents 
                SET {set_clause}
                WHERE id = $1
                RETURNING *
                """,
                *values
            )
            
        if row:
            logger.info("Document updated", document_id=str(document_id))
            return Document(**dict(row))
        return None
    
    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and its embeddings."""
        async with self.pool.acquire() as connection:
            result = await connection.execute(
                "DELETE FROM documents WHERE id = $1",
                document_id
            )
            
        success = result == "DELETE 1"
        if success:
            logger.info("Document deleted", document_id=str(document_id))
        return success
    
    # Embedding operations
    async def create_embedding(self, embedding: EmbeddingCreate) -> Embedding:
        """Create a new embedding."""
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO embeddings (document_id, chunk_text, embedding, chunk_index, chunk_metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, document_id, chunk_text, chunk_index, chunk_metadata, created_at
                """,
                embedding.document_id,
                embedding.chunk_text,
                embedding.embedding,
                embedding.chunk_index,
                embedding.chunk_metadata
            )
            
        logger.info("Embedding created", embedding_id=str(row['id']))
        return Embedding(**dict(row))
    
    async def vector_search(self, request: VectorSearchRequest, query_embedding: List[float]) -> VectorSearchResponse:
        """Perform vector similarity search."""
        start_time = time.time()
        
        # Build the query
        base_query = """
            SELECT 
                e.id as embedding_id,
                e.document_id,
                d.title as document_title,
                e.chunk_text,
                e.chunk_index,
                e.chunk_metadata,
                1 - (e.embedding <=> $1) as similarity_score
            FROM embeddings e
            JOIN documents d ON e.document_id = d.id
            WHERE 1 - (e.embedding <=> $1) >= $2
        """
        
        params = [query_embedding, request.similarity_threshold]
        param_count = 2
        
        # Add document filter if specified
        if request.document_ids:
            param_count += 1
            base_query += f" AND e.document_id = ANY(${param_count})"
            params.append(request.document_ids)
        
        # Add ordering and limit
        base_query += f" ORDER BY similarity_score DESC LIMIT ${param_count + 1}"
        params.append(request.limit)
        
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(base_query, *params)
        
        results = [
            VectorSearchResult(
                embedding_id=row['embedding_id'],
                document_id=row['document_id'],
                document_title=row['document_title'],
                chunk_text=row['chunk_text'],
                chunk_index=row['chunk_index'],
                similarity_score=row['similarity_score'],
                chunk_metadata=row['chunk_metadata']
            )
            for row in rows
        ]
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            "Vector search completed",
            query=request.query,
            results_count=len(results),
            processing_time_ms=processing_time
        )
        
        return VectorSearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time
        )
    
    # Conversation operations
    async def create_conversation(self, conversation: ConversationCreate) -> Conversation:
        """Create a new conversation."""
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO conversations (user_id, title, metadata)
                VALUES ($1, $2, $3)
                RETURNING *
                """,
                conversation.user_id,
                conversation.title,
                conversation.metadata
            )
            
        logger.info("Conversation created", conversation_id=str(row['id']))
        return Conversation(**dict(row))
    
    async def get_conversation(self, conversation_id: UUID) -> Optional[Conversation]:
        """Get a conversation by ID."""
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM conversations WHERE id = $1",
                conversation_id
            )
            
        if row:
            return Conversation(**dict(row))
        return None
    
    async def list_conversations(self, user_id: Optional[UUID] = None, limit: int = 100, offset: int = 0) -> List[Conversation]:
        """List conversations with optional user filter."""
        if user_id:
            query = "SELECT * FROM conversations WHERE user_id = $1 ORDER BY updated_at DESC LIMIT $2 OFFSET $3"
            params = [user_id, limit, offset]
        else:
            query = "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT $1 OFFSET $2"
            params = [limit, offset]
        
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
            
        return [Conversation(**dict(row)) for row in rows]
    
    # Message operations
    async def create_message(self, message: MessageCreate) -> Message:
        """Create a new message."""
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO messages (conversation_id, role, content, metadata, agent_data)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
                """,
                message.conversation_id,
                message.role,
                message.content,
                message.metadata,
                message.agent_data
            )
            
        logger.info("Message created", message_id=str(row['id']))
        return Message(**dict(row))
    
    async def get_conversation_messages(self, conversation_id: UUID, limit: int = 100) -> List[Message]:
        """Get messages for a conversation."""
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM messages 
                WHERE conversation_id = $1 
                ORDER BY created_at ASC 
                LIMIT $2
                """,
                conversation_id, limit
            )
            
        return [Message(**dict(row)) for row in rows]
    
    # Agent log operations
    async def create_agent_log(self, log: AgentLogCreate) -> AgentLog:
        """Create a new agent log entry."""
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO agent_logs (conversation_id, message_id, agent_type, agent_input, 
                                      agent_output, processing_time_ms, status, error_message)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING *
                """,
                log.conversation_id,
                log.message_id,
                log.agent_type,
                log.agent_input,
                log.agent_output,
                log.processing_time_ms,
                log.status,
                log.error_message
            )
            
        return AgentLog(**dict(row))
    
    async def get_agent_logs(self, conversation_id: UUID, agent_type: Optional[str] = None) -> List[AgentLog]:
        """Get agent logs for a conversation."""
        if agent_type:
            query = """
                SELECT * FROM agent_logs 
                WHERE conversation_id = $1 AND agent_type = $2 
                ORDER BY created_at DESC
            """
            params = [conversation_id, agent_type]
        else:
            query = """
                SELECT * FROM agent_logs 
                WHERE conversation_id = $1 
                ORDER BY created_at DESC
            """
            params = [conversation_id]
        
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(query, *params)
            
        return [AgentLog(**dict(row)) for row in rows]


# Global database service instance
db_service = DatabaseService() 
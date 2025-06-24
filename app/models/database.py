"""
Database models for the Agentic RAG AI Agent.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import Field
from .base import BaseAPIModel, IdentifiedModel


class DocumentBase(BaseAPIModel):
    """Base document model."""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    file_type: Optional[str] = Field(None, description="File type (pdf, txt, docx, etc.)")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentCreate(DocumentBase):
    """Model for creating a new document."""
    pass


class DocumentUpdate(BaseAPIModel):
    """Model for updating a document."""
    title: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Document(DocumentBase, IdentifiedModel):
    """Complete document model with ID and timestamps."""
    pass


class EmbeddingBase(BaseAPIModel):
    """Base embedding model."""
    document_id: UUID = Field(..., description="Reference to the source document")
    chunk_text: str = Field(..., description="Text chunk that was embedded")
    chunk_index: int = Field(..., description="Index of the chunk within the document")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk-specific metadata")


class EmbeddingCreate(EmbeddingBase):
    """Model for creating a new embedding."""
    embedding: List[float] = Field(..., description="Vector embedding (1536 dimensions for OpenAI)")


class Embedding(EmbeddingBase, IdentifiedModel):
    """Complete embedding model with ID and timestamps."""
    # Note: embedding vector is not included in API responses for performance
    pass


class EmbeddingWithVector(Embedding):
    """Embedding model including the vector data."""
    embedding: List[float] = Field(..., description="Vector embedding")


class ConversationBase(BaseAPIModel):
    """Base conversation model."""
    user_id: Optional[UUID] = Field(None, description="User ID (when authentication is implemented)")
    title: Optional[str] = Field(None, description="Conversation title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConversationCreate(ConversationBase):
    """Model for creating a new conversation."""
    pass


class ConversationUpdate(BaseAPIModel):
    """Model for updating a conversation."""
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Conversation(ConversationBase, IdentifiedModel):
    """Complete conversation model with ID and timestamps."""
    pass


class MessageBase(BaseAPIModel):
    """Base message model."""
    conversation_id: UUID = Field(..., description="Reference to the conversation")
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    agent_data: Dict[str, Any] = Field(default_factory=dict, description="Agent processing information")


class MessageCreate(MessageBase):
    """Model for creating a new message."""
    pass


class Message(MessageBase, IdentifiedModel):
    """Complete message model with ID and timestamps."""
    pass


class AgentLogBase(BaseAPIModel):
    """Base agent log model."""
    conversation_id: UUID = Field(..., description="Reference to the conversation")
    message_id: Optional[UUID] = Field(None, description="Reference to the related message")
    agent_type: str = Field(..., description="Type of agent (query_rewriter, context_decision, etc.)")
    agent_input: Dict[str, Any] = Field(default_factory=dict, description="Input data for the agent")
    agent_output: Dict[str, Any] = Field(default_factory=dict, description="Output data from the agent")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    status: str = Field(default="success", description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if status is error")


class AgentLogCreate(AgentLogBase):
    """Model for creating a new agent log."""
    pass


class AgentLog(AgentLogBase, IdentifiedModel):
    """Complete agent log model with ID and timestamps."""
    pass


# Search and retrieval models
class VectorSearchRequest(BaseAPIModel):
    """Request model for vector similarity search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    document_ids: Optional[List[UUID]] = Field(None, description="Limit search to specific documents")


class VectorSearchResult(BaseAPIModel):
    """Result model for vector similarity search."""
    embedding_id: UUID = Field(..., description="Embedding ID")
    document_id: UUID = Field(..., description="Source document ID")
    document_title: str = Field(..., description="Source document title")
    chunk_text: str = Field(..., description="Matching text chunk")
    chunk_index: int = Field(..., description="Chunk index within document")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class VectorSearchResponse(BaseAPIModel):
    """Response model for vector similarity search."""
    query: str = Field(..., description="Original search query")
    results: List[VectorSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    processing_time_ms: int = Field(..., description="Search processing time")


# Conversation with messages
class ConversationWithMessages(Conversation):
    """Conversation model including its messages."""
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")


# Document with embeddings count
class DocumentWithStats(Document):
    """Document model with additional statistics."""
    embeddings_count: int = Field(..., description="Number of embeddings for this document")
    total_chunks: int = Field(..., description="Total number of text chunks") 
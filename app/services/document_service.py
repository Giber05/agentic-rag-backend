import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from supabase import Client

from ..core.config import settings
from ..models.document_models import (
    DocumentChunk,
    DocumentResponse,
    ProcessingResult,
    SearchResult,
    DocumentStats
)

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Service for managing documents and embeddings in Supabase database
    """
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    async def store_document(self, processing_result: ProcessingResult, user_id: Optional[str] = None) -> DocumentResponse:
        """
        Store a processed document and its chunks in the database
        
        Args:
            processing_result: Result from document processing
            user_id: Optional user ID for ownership
            
        Returns:
            DocumentResponse with stored document information
        """
        try:
            document_id = str(uuid4())
            current_time = datetime.utcnow()
            
            # Prepare document metadata
            metadata = processing_result.metadata.copy()
            metadata["processing_timestamp"] = current_time.isoformat()
            
            # Store document record
            document_data = {
                "id": document_id,
                "title": processing_result.filename,
                "content": processing_result.text_content,
                "metadata": metadata,
                "created_at": current_time.isoformat(),
                "updated_at": current_time.isoformat(),
                "user_id": user_id  # Can be None, which is fine for the database
            }
            
            # Insert document
            document_result = self.supabase.table("documents").insert(document_data).execute()
            
            if not document_result.data:
                raise Exception("Failed to insert document")
            
            # Store embeddings
            await self._store_embeddings(document_id, processing_result.chunks)
            
            # Create response
            response = DocumentResponse(
                id=document_id,
                filename=processing_result.filename,
                content_type=processing_result.content_type,
                file_size=metadata.get("file_size", 0),
                chunk_count=len(processing_result.chunks),
                created_at=current_time,
                metadata=metadata,
                processing_stats=processing_result.processing_stats
            )
            
            logger.info(f"Successfully stored document {processing_result.filename} with {len(processing_result.chunks)} chunks")
            return response
            
        except Exception as e:
            logger.error(f"Error storing document {processing_result.filename}: {str(e)}")
            raise
    
    async def _store_embeddings(self, document_id: str, chunks: List[DocumentChunk]) -> None:
        """Store document chunks and their embeddings"""
        try:
            embedding_records = []
            
            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                    continue
                
                record = {
                    "id": chunk.id,
                    "document_id": document_id,
                    "chunk_text": chunk.text,
                    "embedding": chunk.embedding,
                    "chunk_index": chunk.chunk_index,
                    "chunk_metadata": chunk.metadata,
                    "created_at": datetime.utcnow().isoformat()
                }
                embedding_records.append(record)
            
            if embedding_records:
                # Insert embeddings in batches to avoid request size limits
                batch_size = 100
                for i in range(0, len(embedding_records), batch_size):
                    batch = embedding_records[i:i + batch_size]
                    result = self.supabase.table("embeddings").insert(batch).execute()
                    
                    if not result.data:
                        raise Exception(f"Failed to insert embedding batch {i//batch_size + 1}")
                
                logger.info(f"Stored {len(embedding_records)} embeddings for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error storing embeddings for document {document_id}: {str(e)}")
            raise
    
    async def get_document(self, document_id: str, user_id: Optional[str] = None) -> Optional[DocumentResponse]:
        """Get a document by ID"""
        try:
            query = self.supabase.table("documents").select("*").eq("id", document_id)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.execute()
            
            if not result.data:
                return None
            
            doc_data = result.data[0]
            
            # Get chunk count
            chunk_count_result = self.supabase.table("embeddings").select("id", count="exact").eq("document_id", document_id).execute()
            chunk_count = chunk_count_result.count or 0
            
            return DocumentResponse(
                id=doc_data["id"],
                filename=doc_data["title"],
                content_type=doc_data["metadata"].get("content_type", "unknown"),
                file_size=doc_data["metadata"].get("file_size", 0),
                chunk_count=chunk_count,
                created_at=datetime.fromisoformat(doc_data["created_at"].replace("Z", "+00:00")),
                metadata=doc_data["metadata"],
                processing_stats=doc_data["metadata"].get("processing_stats", {})
            )
            
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            raise
    
    async def list_documents(
        self,
        user_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        search_query: Optional[str] = None
    ) -> Tuple[List[DocumentResponse], int]:
        """List documents with pagination and optional search"""
        try:
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Build query
            query = self.supabase.table("documents").select("*", count="exact")
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            if search_query:
                query = query.ilike("title", f"%{search_query}%")
            
            # Apply pagination and ordering
            query = query.order("created_at", desc=True).range(offset, offset + page_size - 1)
            
            result = query.execute()
            total_count = result.count or 0
            
            documents = []
            for doc_data in result.data:
                # Get chunk count for each document
                chunk_count_result = self.supabase.table("embeddings").select("id", count="exact").eq("document_id", doc_data["id"]).execute()
                chunk_count = chunk_count_result.count or 0
                
                doc_response = DocumentResponse(
                    id=doc_data["id"],
                    filename=doc_data["title"],
                    content_type=doc_data["metadata"].get("content_type", "unknown"),
                    file_size=doc_data["metadata"].get("file_size", 0),
                    chunk_count=chunk_count,
                    created_at=datetime.fromisoformat(doc_data["created_at"].replace("Z", "+00:00")),
                    metadata=doc_data["metadata"],
                    processing_stats=doc_data["metadata"].get("processing_stats", {})
                )
                documents.append(doc_response)
            
            return documents, total_count
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise
    
    async def delete_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a document and its embeddings"""
        try:
            # First check if document exists and user has permission
            doc_query = self.supabase.table("documents").select("id").eq("id", document_id)
            if user_id:
                doc_query = doc_query.eq("user_id", user_id)
            
            doc_result = doc_query.execute()
            if not doc_result.data:
                return False
            
            # Delete embeddings first (foreign key constraint)
            embeddings_result = self.supabase.table("embeddings").delete().eq("document_id", document_id).execute()
            
            # Delete document
            document_result = self.supabase.table("documents").delete().eq("id", document_id).execute()
            
            if document_result.data:
                logger.info(f"Successfully deleted document {document_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise
    
    async def search_documents(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.5,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search using vector similarity
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results
            threshold: Similarity threshold
            document_ids: Filter by specific document IDs
            user_id: Filter by user ownership
            
        Returns:
            List of search results ordered by similarity
        """
        try:
            start_time = time.time()
            
            # Build the RPC call for vector similarity search
            rpc_params = {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": limit
            }
            
            if document_ids:
                rpc_params["document_ids"] = document_ids
            
            if user_id:
                rpc_params["filter_user_id"] = user_id  # Use filter_user_id to avoid ambiguity
            
            # Call the stored procedure for vector search
            result = self.supabase.rpc("search_embeddings", rpc_params).execute()
            
            search_results = []
            for row in result.data:
                search_result = SearchResult(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    filename=row["filename"],
                    text=row["chunk_text"],
                    similarity_score=row["similarity"],
                    chunk_index=row["chunk_index"],
                    metadata=row.get("metadata", {})
                )
                search_results.append(search_result)
            
            search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            logger.info(f"Vector search completed in {search_time:.2f}ms, found {len(search_results)} results")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
            raise
    
    async def get_document_chunks(
        self,
        document_id: str,
        user_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        try:
            # First verify document access
            doc_query = self.supabase.table("documents").select("id").eq("id", document_id)
            if user_id:
                doc_query = doc_query.eq("user_id", user_id)
            
            doc_result = doc_query.execute()
            if not doc_result.data:
                return []
            
            # Get chunks
            chunks_result = self.supabase.table("embeddings").select("*").eq("document_id", document_id).order("chunk_index").execute()
            
            chunks = []
            for chunk_data in chunks_result.data:
                chunk = DocumentChunk(
                    id=chunk_data["id"],
                    text=chunk_data["chunk_text"],
                    chunk_index=chunk_data["chunk_index"],
                    token_count=len(chunk_data["chunk_text"].split()),  # Approximate
                    embedding=chunk_data["embedding"],
                    metadata=chunk_data.get("chunk_metadata", {})
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {str(e)}")
            raise
    
    async def update_document_metadata(
        self,
        document_id: str,
        metadata: Dict,
        user_id: Optional[str] = None
    ) -> bool:
        """Update document metadata"""
        try:
            query = self.supabase.table("documents").update({
                "metadata": metadata,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", document_id)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.execute()
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Error updating document metadata {document_id}: {str(e)}")
            raise
    
    async def get_document_stats(self, user_id: Optional[str] = None) -> DocumentStats:
        """Get document processing statistics"""
        try:
            # Get document count
            doc_query = self.supabase.table("documents").select("id", count="exact")
            if user_id:
                doc_query = doc_query.eq("user_id", user_id)
            
            doc_result = doc_query.execute()
            total_documents = doc_result.count or 0
            
            # Get chunk count
            chunk_query = self.supabase.table("embeddings").select("id", count="exact")
            if user_id:
                # Join with documents table to filter by user
                chunk_query = chunk_query.join("documents", "document_id", "id").eq("documents.user_id", user_id)
            
            chunk_result = chunk_query.execute()
            total_chunks = chunk_result.count or 0
            
            # Calculate total tokens (approximate)
            total_tokens = total_chunks * 500  # Rough estimate
            
            return DocumentStats(
                total_documents=total_documents,
                total_chunks=total_chunks,
                total_tokens=total_tokens,
                supported_formats=[
                    'text/plain',
                    'application/pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'text/html',
                    'text/markdown'
                ],
                processing_stats={
                    "avg_chunks_per_document": total_chunks / total_documents if total_documents > 0 else 0,
                    "avg_tokens_per_chunk": 500  # Estimate
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            raise 
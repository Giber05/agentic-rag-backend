-- Migration: Add vector search function for semantic search
-- This function performs similarity search using pgvector

CREATE OR REPLACE FUNCTION search_embeddings(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10,
    document_ids text[] DEFAULT NULL,
    user_id uuid DEFAULT NULL
)
RETURNS TABLE (
    chunk_id uuid,
    document_id uuid,
    filename text,
    chunk_text text,
    similarity float,
    chunk_index int,
    metadata jsonb
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id as chunk_id,
        e.document_id,
        d.title as filename,
        e.chunk_text,
        (1 - (e.embedding <=> query_embedding)) as similarity,
        e.chunk_index,
        e.metadata
    FROM embeddings e
    JOIN documents d ON e.document_id = d.id
    WHERE 
        (1 - (e.embedding <=> query_embedding)) > match_threshold
        AND (document_ids IS NULL OR d.id = ANY(document_ids))
        AND (user_id IS NULL OR d.user_id = user_id)
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create an index for better performance on vector operations
CREATE INDEX IF NOT EXISTS embeddings_embedding_idx ON embeddings 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Add comments for documentation
COMMENT ON FUNCTION search_embeddings IS 'Performs semantic search using vector similarity with pgvector';
COMMENT ON INDEX embeddings_embedding_idx IS 'HNSW index for fast vector similarity search'; 
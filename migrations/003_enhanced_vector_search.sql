-- Enhanced vector search function with semantic and keyword search support
-- This migration adds improved search capabilities for the RAG system

-- Drop existing function if it exists
DROP FUNCTION IF EXISTS search_embeddings(vector(1536), float, int, text[], text);

-- Create enhanced search function with multiple search modes
CREATE OR REPLACE FUNCTION search_embeddings(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10,
    document_ids text[] DEFAULT NULL,
    user_id text DEFAULT NULL
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
        (1 - (e.embedding <=> query_embedding)) >= match_threshold
        AND (document_ids IS NULL OR d.id = ANY(document_ids))
        AND (user_id IS NULL OR d.metadata->>'user_id' = user_id OR d.metadata->>'user_id' IS NULL)
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create keyword search function using PostgreSQL full-text search
CREATE OR REPLACE FUNCTION search_keywords(
    search_query text,
    match_count int DEFAULT 10,
    document_ids text[] DEFAULT NULL,
    user_id text DEFAULT NULL
)
RETURNS TABLE (
    chunk_id uuid,
    document_id uuid,
    filename text,
    chunk_text text,
    rank float,
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
        ts_rank(to_tsvector('english', e.chunk_text), plainto_tsquery('english', search_query)) as rank,
        e.chunk_index,
        e.metadata
    FROM embeddings e
    JOIN documents d ON e.document_id = d.id
    WHERE 
        to_tsvector('english', e.chunk_text) @@ plainto_tsquery('english', search_query)
        AND (document_ids IS NULL OR d.id = ANY(document_ids))
        AND (user_id IS NULL OR d.metadata->>'user_id' = user_id OR d.metadata->>'user_id' IS NULL)
    ORDER BY ts_rank(to_tsvector('english', e.chunk_text), plainto_tsquery('english', search_query)) DESC
    LIMIT match_count;
END;
$$;

-- Create hybrid search function that combines semantic and keyword search
CREATE OR REPLACE FUNCTION search_hybrid(
    query_embedding vector(1536),
    search_query text,
    semantic_weight float DEFAULT 0.7,
    keyword_weight float DEFAULT 0.3,
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10,
    document_ids text[] DEFAULT NULL,
    user_id text DEFAULT NULL
)
RETURNS TABLE (
    chunk_id uuid,
    document_id uuid,
    filename text,
    chunk_text text,
    combined_score float,
    semantic_score float,
    keyword_score float,
    chunk_index int,
    metadata jsonb
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH semantic_results AS (
        SELECT 
            e.id as chunk_id,
            e.document_id,
            d.title as filename,
            e.chunk_text,
            (1 - (e.embedding <=> query_embedding)) as semantic_score,
            e.chunk_index,
            e.metadata
        FROM embeddings e
        JOIN documents d ON e.document_id = d.id
        WHERE 
            (document_ids IS NULL OR d.id = ANY(document_ids))
            AND (user_id IS NULL OR d.metadata->>'user_id' = user_id OR d.metadata->>'user_id' IS NULL)
    ),
    keyword_results AS (
        SELECT 
            e.id as chunk_id,
            e.document_id,
            d.title as filename,
            e.chunk_text,
            ts_rank(to_tsvector('english', e.chunk_text), plainto_tsquery('english', search_query)) as keyword_score,
            e.chunk_index,
            e.metadata
        FROM embeddings e
        JOIN documents d ON e.document_id = d.id
        WHERE 
            to_tsvector('english', e.chunk_text) @@ plainto_tsquery('english', search_query)
            AND (document_ids IS NULL OR d.id = ANY(document_ids))
            AND (user_id IS NULL OR d.metadata->>'user_id' = user_id OR d.metadata->>'user_id' IS NULL)
    ),
    combined_results AS (
        SELECT 
            COALESCE(s.chunk_id, k.chunk_id) as chunk_id,
            COALESCE(s.document_id, k.document_id) as document_id,
            COALESCE(s.filename, k.filename) as filename,
            COALESCE(s.chunk_text, k.chunk_text) as chunk_text,
            COALESCE(s.semantic_score, 0.0) * semantic_weight + COALESCE(k.keyword_score, 0.0) * keyword_weight as combined_score,
            COALESCE(s.semantic_score, 0.0) as semantic_score,
            COALESCE(k.keyword_score, 0.0) as keyword_score,
            COALESCE(s.chunk_index, k.chunk_index) as chunk_index,
            COALESCE(s.metadata, k.metadata) as metadata
        FROM semantic_results s
        FULL OUTER JOIN keyword_results k ON s.chunk_id = k.chunk_id
    )
    SELECT 
        cr.chunk_id,
        cr.document_id,
        cr.filename,
        cr.chunk_text,
        cr.combined_score,
        cr.semantic_score,
        cr.keyword_score,
        cr.chunk_index,
        cr.metadata
    FROM combined_results cr
    WHERE cr.combined_score >= match_threshold
    ORDER BY cr.combined_score DESC
    LIMIT match_count;
END;
$$;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_index ON embeddings(chunk_index);
CREATE INDEX IF NOT EXISTS idx_documents_metadata_user_id ON documents USING GIN ((metadata->>'user_id'));

-- Create full-text search index for keyword search
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_text_fts ON embeddings USING GIN (to_tsvector('english', chunk_text));

-- Create composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_embeddings_composite ON embeddings(document_id, chunk_index);

-- Add comments for documentation
COMMENT ON FUNCTION search_embeddings IS 'Semantic vector search using cosine similarity';
COMMENT ON FUNCTION search_keywords IS 'Keyword-based full-text search using PostgreSQL tsvector';
COMMENT ON FUNCTION search_hybrid IS 'Hybrid search combining semantic and keyword search with weighted scoring';

-- Grant execute permissions (adjust as needed for your security model)
-- GRANT EXECUTE ON FUNCTION search_embeddings TO authenticated;
-- GRANT EXECUTE ON FUNCTION search_keywords TO authenticated;
-- GRANT EXECUTE ON FUNCTION search_hybrid TO authenticated; 
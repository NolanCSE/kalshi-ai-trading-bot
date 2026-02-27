-- Supabase pgvector setup for Knowledge Library RAG
-- Run this in the Supabase SQL Editor

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the knowledge chunks table
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id BIGSERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    category TEXT NOT NULL,
    tags JSONB DEFAULT '[]',
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    page_number TEXT,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding 
ON knowledge_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Optional: Index for category filtering
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_category 
ON knowledge_chunks (category);

-- Create RPC function for similarity search with threshold
CREATE OR REPLACE FUNCTION match_knowledge_chunks(
    query_embedding vector(1536),
    match_threshold float,
    match_count int
)
RETURNS TABLE (
    id bigint,
    text text,
    source text,
    source_hash text,
    category text,
    tags jsonb,
    chunk_index int,
    total_chunks int,
    page_number text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        kc.id,
        kc.text,
        kc.source,
        kc.source_hash,
        kc.category,
        kc.tags,
        kc.chunk_index,
        kc.total_chunks,
        kc.page_number,
        1 - (kc.embedding <=> query_embedding) AS similarity
    FROM knowledge_chunks kc
    WHERE 1 - (kc.embedding <=> query_embedding) > match_threshold
    ORDER BY kc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Row Level Security (optional - adjust for your needs)
ALTER TABLE knowledge_chunks ENABLE ROW LEVEL SECURITY;

-- Allow public read access (adjust as needed for your use case)
CREATE POLICY "Allow public read access" ON knowledge_chunks
    FOR SELECT USING (true);

CREATE POLICY "Allow service role insert" ON knowledge_chunks
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow service role delete" ON knowledge_chunks
    FOR DELETE USING (true);

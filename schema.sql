-- Run on a fresh Postgres 16 database with pgvector extension available.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT,
    char_count INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_tenant_idx ON documents (tenant_id);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    char_start INT NOT NULL,
    char_end INT NOT NULL,
    text TEXT NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW for fast vector similarity. m=16, ef_construction=64 are good defaults.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN for full-text BM25 lookups
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Tenant isolation index (every query filters by tenant_id first)
CREATE INDEX IF NOT EXISTS chunks_tenant_doc_idx ON chunks (tenant_id, document_id);

-- Retrieval helper: hybrid score combining vector cosine and BM25 with RRF
CREATE OR REPLACE FUNCTION hybrid_search(
    p_tenant TEXT,
    p_query TEXT,
    p_embedding vector,
    p_limit INT DEFAULT 8
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INT,
    char_start INT,
    char_end INT,
    text TEXT,
    score FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    WITH vec AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> p_embedding) AS rank
        FROM chunks
        WHERE tenant_id = p_tenant
        ORDER BY embedding <=> p_embedding
        LIMIT 30
    ),
    bm AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', p_query)) DESC) AS rank
        FROM chunks
        WHERE tenant_id = p_tenant
          AND tsv @@ plainto_tsquery('english', p_query)
        LIMIT 30
    ),
    fused AS (
        SELECT id, SUM(1.0 / (60 + rank))::FLOAT AS rrf_score
        FROM (
            SELECT id, rank FROM vec
            UNION ALL
            SELECT id, rank FROM bm
        ) u
        GROUP BY id
        ORDER BY rrf_score DESC
        LIMIT p_limit
    )
    SELECT c.id, c.document_id, c.chunk_index, c.char_start, c.char_end, c.text, fused.rrf_score
    FROM fused
    JOIN chunks c ON c.id = fused.id
    ORDER BY fused.rrf_score DESC;
END;
$$;

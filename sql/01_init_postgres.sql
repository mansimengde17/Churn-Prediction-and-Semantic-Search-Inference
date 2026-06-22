-- PostgreSQL initialization script with pgvector
-- Run once on a fresh database

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Customer base table
CREATE TABLE IF NOT EXISTS customers (
    customer_id        VARCHAR(20) PRIMARY KEY,
    gender             VARCHAR(10),
    senior_citizen     SMALLINT   DEFAULT 0,
    partner            SMALLINT   DEFAULT 0,
    dependents         SMALLINT   DEFAULT 0,
    tenure             INTEGER,
    phone_service      SMALLINT   DEFAULT 1,
    internet_service   VARCHAR(20),
    contract           VARCHAR(30),
    monthly_charges    NUMERIC(8,2),
    total_charges      NUMERIC(10,2),
    churn              SMALLINT   DEFAULT 0,
    churn_probability  NUMERIC(5,4),
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Dense embedding store for semantic search
CREATE TABLE IF NOT EXISTS customer_embeddings (
    customer_id    VARCHAR(20) PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    embedding      vector(384),   -- all-MiniLM-L6-v2 output dimension
    customer_text  TEXT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for sub-millisecond approximate nearest-neighbour search
CREATE INDEX IF NOT EXISTS idx_customer_embeddings_hnsw
    ON customer_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Prediction log table (audit trail)
CREATE TABLE IF NOT EXISTS prediction_log (
    id               BIGSERIAL   PRIMARY KEY,
    customer_id      VARCHAR(20),
    churn_probability NUMERIC(5,4),
    risk_level       VARCHAR(10),
    model_version    VARCHAR(20) DEFAULT '1.0.0',
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prediction_log_customer ON prediction_log(customer_id);
CREATE INDEX IF NOT EXISTS idx_prediction_log_created  ON prediction_log(created_at DESC);

-- Convenience view: high-risk customers with their embeddings
CREATE OR REPLACE VIEW v_high_risk_customers AS
SELECT
    c.customer_id,
    c.contract,
    c.internet_service,
    c.monthly_charges,
    c.tenure,
    c.churn_probability,
    1 - (
        ce.embedding <=> (
            SELECT embedding
            FROM customer_embeddings
            ORDER BY embedding <=> ce.embedding
            LIMIT 1 OFFSET 1  -- nearest neighbour (excluding self)
        )
    ) AS nearest_neighbour_similarity
FROM customers c
JOIN customer_embeddings ce USING (customer_id)
WHERE c.churn_probability > 0.70
ORDER BY c.churn_probability DESC;

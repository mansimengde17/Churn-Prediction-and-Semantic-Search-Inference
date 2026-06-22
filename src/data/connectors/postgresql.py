"""
PostgreSQL connector with pgvector support.
Used for storing customer data and embeddings for semantic search.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, func
from loguru import logger

from src.config import settings


engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class CustomerRecord(Base):
    __tablename__ = "customers"

    customer_id = Column(String, primary_key=True)
    gender = Column(String)
    senior_citizen = Column(Integer)
    partner = Column(Integer)
    dependents = Column(Integer)
    tenure = Column(Integer)
    phone_service = Column(Integer)
    internet_service = Column(String)
    contract = Column(String)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    churn = Column(Integer)
    churn_probability = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class PostgreSQLConnector:
    def __init__(self, database_url: str = None):
        url = database_url or settings.DATABASE_URL
        self.engine = create_engine(url, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)

    def initialize_schema(self):
        """Create tables and enable pgvector extension."""
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS customer_embeddings (
                    customer_id VARCHAR PRIMARY KEY,
                    embedding vector(384),
                    customer_text TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_customer_embeddings_hnsw
                ON customer_embeddings USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            conn.commit()
        Base.metadata.create_all(self.engine)
        logger.info("PostgreSQL schema initialized with pgvector")

    def upsert_customers(self, df: pd.DataFrame):
        df.to_sql("customers", self.engine, if_exists="append", index=False, method="multi")
        logger.info(f"Upserted {len(df):,} customer records")

    def fetch_customers(self, limit: int = 1000) -> pd.DataFrame:
        query = f"SELECT * FROM customers LIMIT {limit}"
        return pd.read_sql(query, self.engine)

    def store_embeddings(self, customer_ids: list, embeddings, texts: list):
        with self.Session() as session:
            for cid, emb, txt in zip(customer_ids, embeddings, texts):
                session.execute(
                    text("""
                        INSERT INTO customer_embeddings (customer_id, embedding, customer_text)
                        VALUES (:cid, :emb, :txt)
                        ON CONFLICT (customer_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            customer_text = EXCLUDED.customer_text
                    """),
                    {"cid": cid, "emb": emb.tolist(), "txt": txt},
                )
            session.commit()
        logger.info(f"Stored {len(customer_ids)} embeddings")

    def similarity_search(self, query_embedding, top_k: int = 10) -> pd.DataFrame:
        vec_str = str(query_embedding.tolist())
        sql = f"""
            SELECT customer_id, customer_text,
                   1 - (embedding <=> '{vec_str}'::vector) AS similarity
            FROM customer_embeddings
            ORDER BY embedding <=> '{vec_str}'::vector
            LIMIT {top_k}
        """
        return pd.read_sql(sql, self.engine)

    def run_sql(self, query: str) -> pd.DataFrame:
        return pd.read_sql(query, self.engine)

    def close(self):
        self.engine.dispose()

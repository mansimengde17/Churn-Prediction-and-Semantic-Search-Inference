import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from src.data.generate_synthetic_data import generate_churn_dataset
from src.data.preprocessing import clean_data
from src.features.engineering import add_engineered_features
from src.search.semantic_search import SemanticSearchEngine, customer_to_text


@pytest.fixture(scope="module")
def small_df():
    raw = generate_churn_dataset(n_samples=50)
    clean = clean_data(raw)
    return add_engineered_features(clean)


def test_customer_to_text(small_df):
    row = small_df.iloc[0].to_dict()
    text = customer_to_text(row)
    assert isinstance(text, str)
    assert len(text) > 20
    assert "tenure" in text


def test_embed_shape():
    engine = SemanticSearchEngine()
    texts = ["high risk customer", "loyal long-term customer"]
    emb = engine.embed(texts)
    assert emb.shape == (2, 384)
    # Normalized embeddings should have unit norm
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_and_search(small_df):
    engine = SemanticSearchEngine()
    engine.build_index(small_df)
    results = engine.search("month-to-month fiber optic high monthly charges", top_k=5)
    assert len(results) <= 5
    assert all("customer_id" in r for r in results)
    assert all("similarity_score" in r for r in results)
    # Similarity scores should be descending
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_find_similar(small_df):
    engine = SemanticSearchEngine()
    engine.build_index(small_df)
    cid = engine._customer_ids[0]
    results = engine.find_similar_customers(cid, top_k=3)
    assert len(results) == 3
    ids = [r["customer_id"] for r in results]
    assert cid not in ids  # should not include self

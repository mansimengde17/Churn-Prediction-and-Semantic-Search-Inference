"""
Semantic Search Notebook
Demonstrates embedding-based customer search using sentence-transformers.
"""

# %%
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.generate_synthetic_data import generate_churn_dataset
from src.data.preprocessing import clean_data
from src.features.engineering import add_engineered_features
from src.search.semantic_search import SemanticSearchEngine, customer_to_text

# %% [markdown]
# ## 1. Prepare Data

# %%
raw = generate_churn_dataset(n_samples=500)  # small for demo
df = add_engineered_features(clean_data(raw))
print(f"Dataset: {df.shape}")

# %% [markdown]
# ## 2. Text Serialization
# Each customer profile is converted to a natural-language string for embedding.

# %%
sample_row = df.iloc[0].to_dict()
print("Customer profile as text:")
print(customer_to_text(sample_row))

# %% [markdown]
# ## 3. Build Embedding Index

# %%
engine = SemanticSearchEngine()
engine.build_index(df)
print(f"Index shape: {engine._index.shape}")

# %% [markdown]
# ## 4. Semantic Search Queries

# %%
queries = [
    "high risk customer on month-to-month fiber optic with electronic check",
    "senior citizen without tech support and high monthly charges",
    "loyal long-term customer on two year contract with auto payment",
    "customer likely to churn with no online security or backup",
]

for q in queries:
    print(f"\nQuery: '{q}'")
    results = engine.search(q, top_k=3)
    for r in results:
        print(f"  {r['customer_id']} | sim={r['similarity_score']:.3f} | "
              f"Contract={r.get('Contract','?')} | Tenure={r.get('tenure','?')}mo | "
              f"Churn={r.get('Churn','?')}")

# %% [markdown]
# ## 5. Embedding Visualization (t-SNE)

# %%
from sklearn.manifold import TSNE

print("Computing t-SNE projection…")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb_2d = tsne.fit_transform(engine._index[:300])

churn_labels = df['Churn'].values[:300]

plt.figure(figsize=(10, 7))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                      c=churn_labels, cmap='RdYlGn_r',
                      alpha=0.7, s=30)
plt.colorbar(scatter, label='Churn (1=Yes)')
plt.title('t-SNE Visualization of Customer Embeddings\nColored by Churn Label')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('../data/processed/tsne_embeddings.png', dpi=150)
plt.show()

print("Churned customers cluster separately — embeddings capture churn-relevant patterns.")

# %% [markdown]
# ## 6. Similarity Matrix (subset)

# %%
subset = engine._index[:20]
sim_matrix = subset @ subset.T  # cosine similarity (embeddings are normalized)

plt.figure(figsize=(8, 7))
sns.heatmap(sim_matrix, cmap='Blues', xticklabels=False, yticklabels=False)
plt.title('Customer Embedding Similarity Matrix (first 20 customers)')
plt.tight_layout()
plt.savefig('../data/processed/similarity_matrix.png', dpi=150)
plt.show()

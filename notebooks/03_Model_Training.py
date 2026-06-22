"""
Model Training Notebook
Full training pipeline with cross-validation, hyperparameter tuning, and evaluation.
"""

# %%
import sys
sys.path.insert(0, '..')

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.data.generate_synthetic_data import generate_churn_dataset
from src.data.preprocessing import clean_data, split_data, get_feature_target
from src.features.engineering import add_engineered_features
from src.models.churn_model import build_full_pipeline

plt.style.use('seaborn-v0_8-whitegrid')

# %% [markdown]
# ## 1. Data Preparation

# %%
raw = generate_churn_dataset(n_samples=7043)
clean = clean_data(raw)
engineered = add_engineered_features(clean)

train_df, val_df, test_df = split_data(engineered)
X_train, y_train = get_feature_target(train_df)
X_val, y_val = get_feature_target(val_df)
X_test, y_test = get_feature_target(test_df)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.2%}")

# %% [markdown]
# ## 2. Model Comparison

# %%
models_to_compare = ["lr", "rf", "xgb", "lgbm"]
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name in models_to_compare:
    print(f"Evaluating {name}…", end=" ")
    pipeline = build_full_pipeline(name)
    scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    results[name] = scores
    print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# %%
fig, ax = plt.subplots(figsize=(10, 5))
positions = range(len(results))
for i, (name, scores) in enumerate(results.items()):
    ax.boxplot(scores, positions=[i], widths=0.4)
ax.set_xticklabels(list(results.keys()))
ax.set_ylabel("ROC AUC")
ax.set_title("5-Fold Cross-Validation AUC by Model")
ax.axhline(0.86, color='red', linestyle='--', label='Target: 86%')
ax.legend()
plt.tight_layout()
plt.savefig('../data/processed/model_comparison.png', dpi=150)
plt.show()

# %% [markdown]
# ## 3. Train Production Ensemble

# %%
ensemble = build_full_pipeline("ensemble")
ensemble.fit(X_train, y_train)

train_proba = ensemble.predict_proba(X_train)[:, 1]
val_proba = ensemble.predict_proba(X_val)[:, 1]
test_proba = ensemble.predict_proba(X_test)[:, 1]

print(f"Train AUC:  {roc_auc_score(y_train, train_proba):.4f}")
print(f"Val AUC:    {roc_auc_score(y_val, val_proba):.4f}")
print(f"Test AUC:   {roc_auc_score(y_test, test_proba):.4f}")

# %% [markdown]
# ## 4. ROC & Precision-Recall Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, test_proba)
test_auc = roc_auc_score(y_test, test_proba)
axes[0].plot(fpr, tpr, color='#2196F3', lw=2, label=f'Ensemble (AUC = {test_auc:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, test_proba)
ap = average_precision_score(y_test, test_proba)
axes[1].plot(recall, precision, color='#E91E63', lw=2, label=f'AP = {ap:.4f}')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()

plt.tight_layout()
plt.savefig('../data/processed/roc_pr_curves.png', dpi=150)
plt.show()

# %% [markdown]
# ## 5. Confusion Matrix & Classification Report

# %%
y_pred = (test_proba >= 0.5).astype(int)

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix (threshold=0.5)\nTest AUC: {test_auc:.4f}')
plt.tight_layout()
plt.savefig('../data/processed/confusion_matrix.png', dpi=150)
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

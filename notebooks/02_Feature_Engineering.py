"""
Feature Engineering Notebook
Demonstrates the full feature engineering pipeline.
"""

# %%
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.generate_synthetic_data import generate_churn_dataset
from src.data.preprocessing import clean_data, split_data, get_feature_target
from src.features.engineering import add_engineered_features, build_preprocessor, get_feature_names

plt.style.use('seaborn-v0_8-whitegrid')

# %% [markdown]
# ## 1. Load and Clean Data

# %%
raw = generate_churn_dataset(n_samples=7043)
clean = clean_data(raw)
print(f"Clean shape: {clean.shape}")

# %% [markdown]
# ## 2. Feature Engineering

# %%
engineered = add_engineered_features(clean)
new_features = ['ChargePerTenure', 'NumAddons', 'IsLongTermContract',
                'IsAutoPay', 'IsFiberOptic', 'EngagementScore']

print("New engineered features:")
print(engineered[new_features].describe().round(3))

# %%
# Correlation of engineered features with target
feat_corr = engineered[new_features + ['Churn']].corr()['Churn'].drop('Churn').sort_values()
plt.figure(figsize=(10, 5))
feat_corr.plot(kind='barh', color=['#E91E63' if v > 0 else '#4CAF50' for v in feat_corr])
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Engineered Feature Correlation with Churn')
plt.xlabel('Pearson Correlation')
plt.tight_layout()
plt.savefig('../data/processed/feature_correlation.png', dpi=150)
plt.show()

# %% [markdown]
# ## 3. Preprocessing Pipeline

# %%
train_df, val_df, test_df = split_data(engineered)
X_train, y_train = get_feature_target(train_df)
X_test, y_test = get_feature_target(test_df)

preprocessor = build_preprocessor(include_engineered=True)
X_train_processed = preprocessor.fit_transform(X_train)

print(f"Raw feature matrix: {X_train.shape}")
print(f"Processed feature matrix: {X_train_processed.shape}")

feature_names = get_feature_names(preprocessor)
print(f"\nTotal features after preprocessing: {len(feature_names)}")
print("Sample features:", feature_names[:10])

# %% [markdown]
# ## 4. Feature Importance Preview (using Random Forest)

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_processed, y_train)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 6))
plt.barh(range(len(importance_df)), importance_df['importance'].values)
plt.yticks(range(len(importance_df)), importance_df['feature'].values)
plt.title('Top 20 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../data/processed/feature_importance.png', dpi=150)
plt.show()

print("Top 10 features:")
print(importance_df.head(10)[['feature', 'importance']].to_string(index=False))

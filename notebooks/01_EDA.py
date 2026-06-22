"""
Exploratory Data Analysis — Telco Customer Churn
Run as a script or convert to Jupyter: jupyter nbconvert --to notebook 01_EDA.py
"""

# %% [markdown]
# # Exploratory Data Analysis: Telco Customer Churn
# This notebook explores the churn dataset to understand feature distributions,
# class imbalance, and key churn drivers.

# %%
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.data.generate_synthetic_data import generate_churn_dataset
from src.data.preprocessing import clean_data

plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', 30)

# %% [markdown]
# ## 1. Load Data

# %%
df_raw = generate_churn_dataset(n_samples=7043)
print(f"Shape: {df_raw.shape}")
print(f"Churn rate: {df_raw['Churn'].mean():.2%}")
df_raw.head()

# %%
df_raw.info()

# %%
df_raw.describe()

# %% [markdown]
# ## 2. Target Variable Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Churn count
churn_counts = df_raw['Churn'].value_counts()
axes[0].bar(['No Churn', 'Churn'], churn_counts, color=['#4CAF50', '#E91E63'])
axes[0].set_title('Class Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate(churn_counts):
    axes[0].text(i, v + 30, f'{v} ({v/len(df_raw):.1%})', ha='center')

# Monthly charges by churn
df_raw.groupby('Churn')['MonthlyCharges'].plot(kind='kde', ax=axes[1])
axes[1].set_title('Monthly Charges Distribution by Churn')
axes[1].legend(['No Churn', 'Churn'])

plt.tight_layout()
plt.savefig('../data/processed/eda_target.png', dpi=150)
plt.show()

# %% [markdown]
# ## 3. Feature Analysis

# %%
# Churn rate by Contract type
contract_churn = df_raw.groupby('Contract')['Churn'].agg(['mean', 'count']).reset_index()
contract_churn.columns = ['Contract', 'Churn Rate', 'Count']
contract_churn['Churn Rate %'] = (contract_churn['Churn Rate'] * 100).round(1)
print("\nChurn rate by contract:")
print(contract_churn)

# %%
# Churn rate by Internet Service
print("\nChurn rate by Internet Service:")
print(df_raw.groupby('InternetService')['Churn'].mean().round(4))

# %%
# Churn rate by Payment Method
print("\nChurn rate by Payment Method:")
print(df_raw.groupby('PaymentMethod')['Churn'].mean().round(4))

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Contract vs Churn
contract_pivot = df_raw.groupby('Contract')['Churn'].mean() * 100
axes[0, 0].bar(contract_pivot.index, contract_pivot.values, color=['#E91E63', '#FF9800', '#4CAF50'])
axes[0, 0].set_title('Churn Rate by Contract Type')
axes[0, 0].set_ylabel('Churn Rate (%)')

# Tenure distribution by churn
df_raw[df_raw['Churn'] == 0]['tenure'].hist(bins=30, ax=axes[0, 1], alpha=0.7, label='No Churn', color='#4CAF50')
df_raw[df_raw['Churn'] == 1]['tenure'].hist(bins=30, ax=axes[0, 1], alpha=0.7, label='Churn', color='#E91E63')
axes[0, 1].set_title('Tenure Distribution')
axes[0, 1].legend()

# Internet Service vs Churn
internet_churn = df_raw.groupby('InternetService')['Churn'].mean() * 100
axes[1, 0].bar(internet_churn.index, internet_churn.values, color=['#2196F3', '#9C27B0', '#757575'])
axes[1, 0].set_title('Churn Rate by Internet Service')

# Monthly charges scatter vs tenure
axes[1, 1].scatter(df_raw[df_raw['Churn']==0]['tenure'], df_raw[df_raw['Churn']==0]['MonthlyCharges'],
                   alpha=0.3, c='#4CAF50', label='No Churn', s=10)
axes[1, 1].scatter(df_raw[df_raw['Churn']==1]['tenure'], df_raw[df_raw['Churn']==1]['MonthlyCharges'],
                   alpha=0.3, c='#E91E63', label='Churn', s=10)
axes[1, 1].set_xlabel('Tenure (months)')
axes[1, 1].set_ylabel('Monthly Charges ($)')
axes[1, 1].set_title('Tenure vs Monthly Charges')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('../data/processed/eda_features.png', dpi=150)
plt.show()

# %% [markdown]
# ## 4. Correlation Analysis

# %%
df_clean = clean_data(df_raw)
numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
corr = df_clean[numeric_cols].corr()

plt.figure(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            mask=mask, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('../data/processed/eda_correlation.png', dpi=150)
plt.show()

# %% [markdown]
# ## 5. Key Insights
#
# 1. **Class imbalance**: ~26% churn rate — manageable with class_weight='balanced'
# 2. **Contract type is the strongest driver**: Month-to-month customers churn at 40%+ vs ~10% on 2-year
# 3. **Fiber optic customers churn more**: Likely due to higher costs and more alternatives
# 4. **Electronic check payment correlates strongly with churn**
# 5. **Tenure is negatively correlated with churn** — long-tenured customers are loyal
# 6. **Monthly charges are positively correlated** — higher bills → more churn risk
print("EDA complete. See data/processed/ for saved visualizations.")

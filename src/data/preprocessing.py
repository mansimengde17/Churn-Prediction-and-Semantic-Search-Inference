"""
Data loading and preprocessing for the churn pipeline.
Handles cleaning, type casting, and train/test splitting.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger

from src.config import settings


# Only Churn (the target) is pre-converted to int.
# Partner, Dependents, PhoneService, PaperlessBilling stay as "Yes"/"No" strings —
# the ColumnTransformer OrdinalEncoder handles them in a type-safe way.
BINARY_COLS = ["Churn"]

# All feature columns (customerID excluded)
CATEGORICAL_COLS = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

NUMERICAL_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"


def load_raw_data(path: str | Path = None) -> pd.DataFrame:
    if path is None:
        path = settings.RAW_DATA_PATH / "telco_churn.csv"
    path = Path(path)
    if not path.exists():
        logger.info("Raw data not found — generating synthetic dataset…")
        from src.data.generate_synthetic_data import generate_churn_dataset
        generate_churn_dataset(output_path=str(path))
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is sometimes stored as string with spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill TotalCharges NaN with tenure * MonthlyCharges (new customers)
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]

    # Binary encoding
    for col in BINARY_COLS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Drop rows with any remaining nulls in key columns
    before = len(df)
    df = df.dropna(subset=NUMERICAL_COLS + [TARGET_COL])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with nulls")

    return df


def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train_df, val_df, test_df)."""
    feature_cols = [c for c in df.columns if c not in ["customerID", target]]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_relative, stratify=y_train, random_state=random_state
    )

    train_df = X_train.copy()
    train_df[target] = y_train.values

    val_df = X_val.copy()
    val_df[target] = y_val.values

    test_df = X_test.copy()
    test_df[target] = y_test.values

    logger.info(
        f"Split → train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )
    return train_df, val_df, test_df


def get_feature_target(df: pd.DataFrame, target: str = TARGET_COL):
    feature_cols = [c for c in df.columns if c not in ["customerID", target]]
    return df[feature_cols], df[target]

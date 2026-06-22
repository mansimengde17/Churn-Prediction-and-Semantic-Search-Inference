"""
Feature engineering for the churn prediction pipeline.
Builds a scikit-learn ColumnTransformer that handles encoding + scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer

# ── Column groups ─────────────────────────────────────────────────────────────

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

# Yes/No binary columns — handled by OrdinalEncoder(categories=[["No","Yes"]])
# so they work regardless of whether upstream cleaning already converted them
# to 0/1 integers or left them as "Yes"/"No" strings.
BINARY_FEATURES = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

ORDINAL_FEATURES = ["Contract"]
ORDINAL_CATEGORIES = [["Month-to-month", "One year", "Two year"]]

NOMINAL_FEATURES = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaymentMethod",
]

ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features (pure pandas transformation)."""
    df = df.copy()

    # Unit economics: monthly cost per tenure month
    df["ChargePerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Count of subscribed add-on services
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumAddons"] = sum(
        (df[col] == "Yes").astype(int)
        for col in addon_cols
        if col in df.columns
    )

    # Contract durability (0 = month-to-month, highest risk)
    df["IsLongTermContract"] = (df["Contract"] != "Month-to-month").astype(int)

    # Auto-payment indicator (lower churn risk)
    df["IsAutoPay"] = df["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)

    # High-cost internet flag
    df["IsFiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)

    # Composite engagement score
    df["EngagementScore"] = (
        df["NumAddons"] * 0.2
        + df["tenure"] * 0.05
        + df["IsLongTermContract"] * 1.5
    )

    return df


def build_preprocessor(include_engineered: bool = True) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer preprocessing pipeline."""

    num_cols = NUMERICAL_FEATURES.copy()
    if include_engineered:
        num_cols += [
            "ChargePerTenure", "NumAddons", "EngagementScore",
            "IsLongTermContract", "IsAutoPay", "IsFiberOptic",
        ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # OrdinalEncoder handles both "Yes"/"No" strings and 0/1 integers robustly.
    # Using string categories ensures consistent encoding regardless of upstream cleaning.
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["No", "Yes"]] * len(BINARY_FEATURES),
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=float,
        )),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=float,
        )),
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, num_cols),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
            ("nominal", nominal_pipeline, NOMINAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names after fitting the preprocessor."""
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "nominal":
            enc = transformer.named_steps["encoder"]
            names.extend(enc.get_feature_names_out(cols).tolist())
        elif isinstance(cols, list):
            names.extend(cols)
        else:
            names.append(cols)
    return names

"""
Churn prediction model definitions.
Provides Random Forest, XGBoost, LightGBM, and a Voting Ensemble.
The ensemble is the production model targeting 86%+ AUC.
"""

from __future__ import annotations

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.features.engineering import build_preprocessor, add_engineered_features


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=1,   # 1 avoids nested-parallelism deadlock inside VotingClassifier
        random_state=random_state,
    )


def build_xgboost(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=2.5,
        eval_metric="auc",
        random_state=random_state,
        n_jobs=1,
    )


def build_lightgbm(random_state: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.08,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.05,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )


def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )


def build_ensemble(random_state: int = 42) -> VotingClassifier:
    # n_jobs=1 here: individual estimators already use n_jobs=-1 internally;
    # VotingClassifier's own parallelism triggers loky subprocess pickling of
    # mixed-dtype arrays which causes dtype coercion errors.
    return VotingClassifier(
        estimators=[
            ("rf", build_random_forest(random_state)),
            ("xgb", build_xgboost(random_state)),
            ("lgbm", build_lightgbm(random_state)),
            ("lr", build_logistic_regression()),
        ],
        voting="soft",
        weights=[2, 3, 3, 1],
        n_jobs=1,
    )


def build_full_pipeline(model_type: str = "ensemble", random_state: int = 42) -> Pipeline:
    """
    Returns a complete sklearn Pipeline:
        feature_engineering → preprocessing → model
    The pipeline's predict / predict_proba can be called directly on raw DataFrames.
    """
    model_map = {
        "rf": build_random_forest(random_state),
        "xgb": build_xgboost(random_state),
        "lgbm": build_lightgbm(random_state),
        "lr": build_logistic_regression(),
        "ensemble": build_ensemble(random_state),
    }
    if model_type not in model_map:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(model_map)}")

    preprocessor = build_preprocessor(include_engineered=True)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model_map[model_type]),
    ])

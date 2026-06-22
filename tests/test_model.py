import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.data.generate_synthetic_data import generate_churn_dataset
from src.data.preprocessing import clean_data, split_data, get_feature_target
from src.features.engineering import add_engineered_features
from src.models.churn_model import build_full_pipeline


@pytest.fixture(scope="module")
def prepared_data():
    raw = generate_churn_dataset(n_samples=1000)
    clean = clean_data(raw)
    engineered = add_engineered_features(clean)
    train, val, test = split_data(engineered)
    X_train, y_train = get_feature_target(train)
    X_test, y_test = get_feature_target(test)
    return X_train, y_train, X_test, y_test


def test_logistic_regression_pipeline(prepared_data):
    X_train, y_train, X_test, y_test = prepared_data
    pipeline = build_full_pipeline("lr")
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    assert auc >= 0.70, f"LR AUC too low: {auc:.4f}"


def test_random_forest_pipeline(prepared_data):
    X_train, y_train, X_test, y_test = prepared_data
    pipeline = build_full_pipeline("rf")
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    assert auc >= 0.78, f"RF AUC too low: {auc:.4f}"


def test_xgboost_pipeline(prepared_data):
    X_train, y_train, X_test, y_test = prepared_data
    pipeline = build_full_pipeline("xgb")
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    assert auc >= 0.80, f"XGBoost AUC too low: {auc:.4f}"


def test_predict_proba_shape(prepared_data):
    X_train, y_train, X_test, _ = prepared_data
    pipeline = build_full_pipeline("lr")
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_labels_binary(prepared_data):
    X_train, y_train, X_test, _ = prepared_data
    pipeline = build_full_pipeline("lr")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert set(preds).issubset({0, 1})

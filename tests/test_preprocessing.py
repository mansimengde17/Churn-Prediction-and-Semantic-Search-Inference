import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest

from src.data.generate_synthetic_data import generate_churn_dataset
from src.data.preprocessing import clean_data, split_data, get_feature_target
from src.features.engineering import add_engineered_features


@pytest.fixture(scope="module")
def raw_df():
    return generate_churn_dataset(n_samples=500)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)


def test_dataset_shape(raw_df):
    assert raw_df.shape == (500, 21)
    assert "Churn" in raw_df.columns
    assert "customerID" in raw_df.columns


def test_churn_rate(raw_df):
    rate = raw_df["Churn"].mean()
    assert 0.10 <= rate <= 0.45, f"Unexpected churn rate: {rate:.2%}"


def test_clean_data_no_nulls(clean_df):
    assert clean_df["TotalCharges"].isna().sum() == 0
    assert clean_df["MonthlyCharges"].isna().sum() == 0


def test_binary_encoding(clean_df):
    # Churn (target) is converted to 0/1 int
    assert set(clean_df["Churn"].unique()).issubset({0, 1})
    # Feature binary cols stay as "Yes"/"No" — OrdinalEncoder handles them
    assert set(clean_df["Partner"].unique()).issubset({"Yes", "No"})


def test_split_sizes(clean_df):
    train, val, test = split_data(clean_df, test_size=0.2, val_size=0.1)
    total = len(train) + len(val) + len(test)
    assert total == len(clean_df)
    assert len(test) == pytest.approx(len(clean_df) * 0.2, abs=5)


def test_engineered_features(clean_df):
    df = add_engineered_features(clean_df)
    assert "ChargePerTenure" in df.columns
    assert "NumAddons" in df.columns
    assert "EngagementScore" in df.columns
    assert "IsLongTermContract" in df.columns
    assert df["NumAddons"].between(0, 6).all()
    assert df["IsLongTermContract"].isin([0, 1]).all()

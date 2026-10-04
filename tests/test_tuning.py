"""Tuning: time-aware folds never validate on the past, and parameters stay in range."""
import numpy as np
import pandas as pd
from src.credit_default.tune import time_folds, sample_params


def test_folds_validate_only_on_later_loans():
    df = pd.DataFrame({"issue_date": pd.date_range("2012-01-01", periods=36, freq="MS")})
    for train, valid in time_folds(df):
        assert len(train) > 0 and len(valid) > 0
        assert train["issue_date"].max() < valid["issue_date"].min()


def test_sampled_params_in_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = sample_params(rng)
        assert 2 <= p["max_depth"] <= 6
        assert 0 < p["learning_rate"] <= 0.1
        assert 0.5 <= p["subsample"] <= 1.0

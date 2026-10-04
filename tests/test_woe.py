"""WoE scorecard: WoE signs, information value, unseen categories, and score scaling."""
import numpy as np
import pandas as pd
import pytest
from src.credit_default.woe import WoEBinner, score_from_pd, BASE_SCORE, BASE_ODDS, PDO


def toy_data(bad_a=5, bad_b=30):
    X = pd.DataFrame({"grade": ["A"] * 100 + ["B"] * 100})
    y = pd.Series([0] * (100 - bad_a) + [1] * bad_a + [0] * (100 - bad_b) + [1] * bad_b)
    return X, y


def test_woe_positive_for_safer_bin_and_iv_high():
    X, y = toy_data()
    b = WoEBinner().fit(X, y)
    assert b.woe_["grade"]["A"] > 0 > b.woe_["grade"]["B"]
    assert b.iv_["grade"] > 0.1


def test_iv_near_zero_when_unrelated():
    X, y = toy_data(bad_a=20, bad_b=20)
    assert WoEBinner().fit(X, y).iv_["grade"] == pytest.approx(0.0, abs=1e-3)


def test_unseen_category_gets_neutral_woe():
    X, y = toy_data()
    b = WoEBinner().fit(X, y)
    out = b.transform(pd.DataFrame({"grade": ["Z"]}))
    assert out["grade"].iloc[0] == 0.0


def test_numeric_values_outside_training_range_are_binned():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=1_000)})
    y = pd.Series((rng.random(1_000) < 0.2).astype(int))
    b = WoEBinner().fit(X, y)
    out = b.transform(pd.DataFrame({"x": [-100.0, 100.0]}))
    assert np.isfinite(out["x"]).all()


def test_score_scaling():
    base_pd = 1 / (1 + BASE_ODDS)                 # odds of 50:1 against default
    double_pd = 1 / (1 + 2 * BASE_ODDS)           # odds doubled
    assert score_from_pd(np.array([base_pd]))[0] == pytest.approx(BASE_SCORE)
    assert (score_from_pd(np.array([double_pd]))[0]
            - score_from_pd(np.array([base_pd]))[0]) == pytest.approx(PDO)

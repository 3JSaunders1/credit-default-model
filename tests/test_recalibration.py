"""Recalibration: the intercept shift hits its target and preserves ranking."""
import numpy as np
import pandas as pd
import pytest
from src.credit_default.recalibrate import (intercept_shift, apply_shift,
                                            split_for_recalibration)

rng = np.random.default_rng(0)
P = rng.uniform(0.02, 0.40, size=5_000)


def test_shift_hits_target_rate():
    b = intercept_shift(P, 0.20)
    assert apply_shift(P, b).mean() == pytest.approx(0.20, abs=1e-6)


def test_shift_preserves_ranking():
    adjusted = apply_shift(P, intercept_shift(P, 0.25))
    assert (np.argsort(P) == np.argsort(adjusted)).all()


def test_no_shift_when_already_calibrated():
    assert intercept_shift(P, P.mean()) == pytest.approx(0.0, abs=1e-6)


def test_probabilities_stay_in_bounds():
    adjusted = apply_shift(P, 3.0)
    assert ((adjusted > 0) & (adjusted < 1)).all()


def test_recalibration_holdout_is_separate():
    df = pd.DataFrame({"issue_date": pd.date_range("2012-01-01", periods=36, freq="MS")})
    fit, holdout = split_for_recalibration(df)
    assert fit["issue_date"].max() < pd.Timestamp("2014-01-01")
    assert holdout["issue_date"].min() >= pd.Timestamp("2014-01-01")
    assert len(fit) + len(holdout) == len(df)
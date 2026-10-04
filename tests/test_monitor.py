"""Monitoring: categorical PSI and traffic-light thresholds."""
import pandas as pd
import pytest
from src.credit_default.monitor import categorical_psi, feature_psi_value, psi_status


def test_categorical_psi_zero_for_same_mix():
    s = pd.Series(["RENT"] * 50 + ["MORTGAGE"] * 50)
    assert categorical_psi(s, s) == pytest.approx(0.0, abs=1e-9)


def test_categorical_psi_flags_shift():
    before = pd.Series(["RENT"] * 80 + ["MORTGAGE"] * 20)
    after = pd.Series(["RENT"] * 20 + ["MORTGAGE"] * 80)
    assert categorical_psi(before, after) > 0.25


def test_binary_features_use_category_shares():
    before = pd.Series([0] * 90 + [1] * 10)
    after = pd.Series([0] * 50 + [1] * 50)
    assert feature_psi_value(before, after) > 0.25   # binned PSI would miss this


def test_psi_status_thresholds():
    assert psi_status(0.05) == "green"
    assert psi_status(0.15) == "amber"
    assert psi_status(0.30) == "red"

def test_text_features_use_category_shares():
    before = pd.Series(["RENT"] * 80 + ["MORTGAGE"] * 20, dtype="string")
    after = pd.Series(["RENT"] * 20 + ["MORTGAGE"] * 80, dtype="string")
    assert feature_psi_value(before, after) > 0.25

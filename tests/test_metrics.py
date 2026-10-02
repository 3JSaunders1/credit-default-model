"""Metric correctness: PSI, KS, AUC/Gini, and the calibration table."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from src.credit_default.evaluate import psi, ks_stat, calibration_table

rng = np.random.default_rng(42)


def test_psi_is_zero_for_identical_distributions():
    x = rng.normal(size=10_000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_flags_a_large_shift():
    x = rng.normal(size=10_000)
    assert psi(x, x + 1.0) > 0.25          # "significant shift" threshold


def test_psi_is_non_negative():
    x, y = rng.normal(size=5_000), rng.normal(0.2, 1.1, size=5_000)
    assert psi(x, y) >= 0


def test_ks_is_one_for_perfect_separation():
    y = np.array([0] * 50 + [1] * 50)
    assert ks_stat(y, y.astype(float)) == pytest.approx(1.0)


def test_ks_is_near_zero_for_random_scores():
    y = rng.integers(0, 2, size=20_000)
    assert ks_stat(y, rng.random(20_000)) < 0.05


def test_auc_and_gini_for_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    auc = roc_auc_score(y, [0.1, 0.2, 0.8, 0.9])
    assert auc == 1.0 and 2 * auc - 1 == 1.0


def test_calibration_table_counts_and_order():
    p = rng.random(1_000)
    y = (rng.random(1_000) < p).astype(int)
    table = calibration_table(y, p, bins=10)
    assert len(table) == 10
    assert table["loans"].sum() == 1_000
    assert table["predicted"].is_monotonic_increasing
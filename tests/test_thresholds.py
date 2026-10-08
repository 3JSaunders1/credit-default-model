"""Threshold analysis: confusion counts, metrics, and policy selection."""
import numpy as np
import pytest
from src.credit_default.thresholds import best_policy, confusion_at, threshold_table


def test_counts_add_up():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 500), rng.random(500)
    assert sum(confusion_at(y, p, 0.3)) == 500


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.10, 0.15, 0.80, 0.90])
    row = threshold_table(y, p, thresholds=[0.5]).iloc[0]
    assert row["precision"] == pytest.approx(1.0) and row["recall"] == pytest.approx(1.0)
    assert row["bad_rate_approved"] == pytest.approx(0.0)


def test_approval_rate_rises_with_threshold():
    rng = np.random.default_rng(1)
    table = threshold_table(rng.integers(0, 2, 1000), rng.random(1000))
    assert table["approval_rate"].is_monotonic_increasing


def test_policy_respects_cap():
    rng = np.random.default_rng(2)
    p = rng.random(2000)
    y = (rng.random(2000) < p * 0.4).astype(int)
    policy = best_policy(threshold_table(y, p), max_bad_rate=0.10)
    assert policy is not None and policy["bad_rate_approved"] <= 0.10

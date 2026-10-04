"""Expected loss: LGD and EAD ratios stay in bounds, and EL = PD x LGD x EAD."""
import numpy as np
import pandas as pd
import pytest
from src.credit_default.expected_loss import loss_components, expected_loss


def toy_losses():
    return pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "funded_amnt": [10_000, 10_000, 10_000, 10_000],
        "ead": [6_000, 6_000, 4_000, 0],               # "d" has nothing outstanding
        "net_recovery": [6_000, 0, 1_000, 0],
    })


def test_full_and_zero_recovery():
    out = loss_components(toy_losses()).set_index("id")
    assert out.loc["a", "lgd"] == pytest.approx(0.0)   # fully recovered
    assert out.loc["b", "lgd"] == pytest.approx(1.0)   # nothing recovered
    assert out.loc["c", "lgd"] == pytest.approx(0.75)


def test_zero_exposure_loans_dropped():
    assert "d" not in set(loss_components(toy_losses())["id"])


def test_ratios_bounded_and_loss_consistent():
    out = loss_components(toy_losses())
    assert out["lgd"].between(0, 1).all() and out["ead_ratio"].between(0, 1).all()
    assert np.allclose(out["loss"], out["ead"] * out["lgd"])


def test_expected_loss_formula():
    el = expected_loss(np.array([0.1, 0.2]), 0.8, np.array([5_000, 10_000]))
    assert np.allclose(el, [400, 1_600])

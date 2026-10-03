"""Shared preprocessing: every script uses the same definition."""
from src.credit_default.train import NUMERIC, CATEGORICAL, make_preprocessor, make_logit


def test_default_columns():
    prep = make_preprocessor()
    assert prep.transformers[0][2] == NUMERIC
    assert prep.transformers[1][2] == CATEGORICAL


def test_custom_columns_keep_same_settings():
    base = make_preprocessor()
    extended = make_preprocessor(NUMERIC + ["state_unemp"])
    assert "state_unemp" in extended.transformers[0][2]
    # Encoder settings must be identical, so comparisons stay apples to apples
    assert (base.transformers[1][1].get_params()
            == extended.transformers[1][1].get_params())


def test_logit_uses_shared_preprocessing():
    model = make_logit()
    assert list(model.named_steps) == ["prep", "model"]

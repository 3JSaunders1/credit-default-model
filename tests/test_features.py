"""Feature engineering: parsing, missing indicators, and safe ratios."""
import numpy as np
import pandas as pd
from src.credit_default.features import clean


def make_df():
    return pd.DataFrame({
        "emp_length": ["10+ years", "< 1 year", "3 years", None],
        "annual_inc": [50_000, 0, 80_000, 60_000],
        "loan_amnt": [10_000, 5_000, 20_000, 15_000],
        "dti": [15.0, None, 30.0, 10.0],
        "revol_util": [50.0, 20.0, None, 80.0],
    })


def test_employment_length_parsing():
    out = clean(make_df())
    assert out["emp_length_yrs"].tolist()[:3] == [10.0, 0.0, 3.0]
    assert np.isnan(out["emp_length_yrs"].iloc[3])


def test_missing_indicators():
    out = clean(make_df())
    assert out["dti_missing"].tolist() == [0, 1, 0, 0]
    assert out["revol_util_missing"].tolist() == [0, 0, 1, 0]
    assert out["emp_length_yrs_missing"].tolist() == [0, 0, 0, 1]


def test_loan_to_income_handles_zero_income():
    out = clean(make_df())
    assert np.isnan(out["loan_to_income"].iloc[1])       # no division by zero
    assert np.isfinite(out["loan_to_income"].drop(index=1)).all()


def test_does_not_modify_input():
    df = make_df()
    clean(df)
    assert "emp_length_yrs" not in df.columns
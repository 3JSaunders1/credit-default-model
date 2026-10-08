"""Clean data, engineer features, and create the out-of-time split."""
import numpy as np
import pandas as pd
from src.credit_default.config import (DATA_DIR, TRAIN_START, TRAIN_END, TEST_START, TEST_END)

NUMERIC = ["loan_amnt", "annual_inc", "dti", "fico_range_low", "revol_util",
           "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec"]
CATEGORICAL = ["home_ownership", "purpose"]


from src.credit_default.logging_utils import get_logger, run_main

log = get_logger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # "10+ years" -> 10, "< 1 year" -> 0, missing stays missing
    df["emp_length_yrs"] = (df["emp_length"]
                            .str.replace("< 1", "0", regex=False)
                            .str.extract(r"(\d+)")[0]
                            .astype(float))

    # Missing indicators, since missingness itself can predict risk
    for col in ["emp_length_yrs", "dti", "revol_util"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Skewed income: log-transform, and cap extreme DTI values
    df["log_annual_inc"] = np.log1p(df["annual_inc"])
    df["dti"] = df["dti"].clip(upper=df["dti"].quantile(0.99))

    # Ratio feature: loan size relative to income
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    return df


def split(df: pd.DataFrame):
    train = df[(df["issue_date"] >= TRAIN_START) & (df["issue_date"] <= TRAIN_END)]
    test = df[(df["issue_date"] >= TEST_START) & (df["issue_date"] <= TEST_END)]
    return train, test


def main():
    df = pd.read_parquet(DATA_DIR / "processed" / "loans.parquet")
    df = clean(df)
    train, test = split(df)

    # Impute with TRAINING medians only, to avoid leakage
    fill_cols = NUMERIC + ["emp_length_yrs", "loan_to_income", "log_annual_inc"]
    medians = train[fill_cols].median()
    train = train.fillna(medians)
    test = test.fillna(medians)

    train.to_parquet(DATA_DIR / "processed" / "train.parquet")
    test.to_parquet(DATA_DIR / "processed" / "test.parquet")

    print(f"Train: {len(train):,} loans, default rate {train['default_flag'].mean():.1%}")
    print(f"Test:  {len(test):,} loans, default rate {test['default_flag'].mean():.1%}")


if __name__ == "__main__":
    run_main(main)
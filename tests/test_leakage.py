"""Leakage checks: no post-origination fields, no Lending Club model outputs, clean time split."""
import pandas as pd
from src.credit_default.config import ROOT, TRAIN_END, TEST_START
from src.credit_default.features import split
from src.credit_default.train import NUMERIC, CATEGORICAL, TARGET

# Fields only known after a loan plays out
POST_ORIGINATION = {
    "total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "last_credit_pull_d", "out_prncp", "debt_settlement_flag", "settlement_status",
}
# Outputs of Lending Club's own risk model
LENDING_CLUB_MODEL = {"grade", "sub_grade", "int_rate"}

FEATURES = set(NUMERIC + CATEGORICAL)


def test_no_post_origination_features():
    assert not FEATURES & POST_ORIGINATION


def test_lending_club_model_outputs_excluded():
    assert not FEATURES & LENDING_CLUB_MODEL


def test_target_is_not_a_feature():
    assert TARGET not in FEATURES


def test_sql_selects_no_post_origination_fields():
    sql = (ROOT / "sql" / "01_build_loan_table.sql").read_text().lower()
    for col in POST_ORIGINATION:
        assert col not in sql, f"Leakage column in SQL: {col}"


def test_train_period_ends_before_test_period():
    assert pd.Timestamp(TRAIN_END) < pd.Timestamp(TEST_START)


def test_split_has_no_time_overlap():
    df = pd.DataFrame({
        "id": range(48),
        "issue_date": pd.date_range("2012-01-01", periods=48, freq="MS"),
    })
    train, test = split(df)
    assert train["issue_date"].max() < test["issue_date"].min()
    assert not set(train["id"]) & set(test["id"])
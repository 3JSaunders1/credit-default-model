"""Spark pipeline: filters and target definition on a tiny synthetic file."""
import pytest

pyspark = pytest.importorskip("pyspark")
from src.credit_default.spark_pipeline import get_spark, build_loans


def test_filters_and_default_flag(tmp_path):
    csv = tmp_path / "loans.csv"
    csv.write_text(
        "id,loan_status,term,issue_d,loan_amnt\n"
        "1,Fully Paid, 36 months,Jan-2014,1000\n"
        "2,Charged Off, 36 months,Feb-2014,2000\n"
        "3,Current, 36 months,Mar-2014,3000\n"     # dropped: no final outcome
        "4,Charged Off, 60 months,Apr-2014,4000\n"  # dropped: 60-month term
    )
    spark = get_spark()
    rows = build_loans(spark, str(csv)).orderBy("id").collect()
    assert [r["id"] for r in rows] == ["1", "2"]
    assert [r["default_flag"] for r in rows] == [0, 1]
    assert str(rows[0]["issue_date"]) == "2014-01-01"

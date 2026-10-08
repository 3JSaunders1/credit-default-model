"""Parity check: do the DuckDB and Spark pipelines produce the same loan table?"""
import pandas as pd
from src.credit_default.config import DATA_DIR


from src.credit_default.logging_utils import get_logger, run_main

log = get_logger(__name__)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(year=pd.to_datetime(df["issue_date"]).dt.year)
    return df.groupby("year").agg(loans=("default_flag", "size"),
                                  default_rate=("default_flag", "mean"))


def main():
    duck = pd.read_parquet(DATA_DIR / "processed" / "loans.parquet",
                           columns=["issue_date", "default_flag"])
    spark = pd.read_parquet(DATA_DIR / "processed" / "loans_spark.parquet",
                            columns=["issue_date", "default_flag"])

    table = summarize(duck).join(summarize(spark), lsuffix="_duckdb", rsuffix="_spark")
    table["loan_diff"] = table["loans_spark"] - table["loans_duckdb"]
    table["rate_diff_pp"] = (table["default_rate_spark"] - table["default_rate_duckdb"]) * 100
    print(table.round(4).to_string())

    total_diff = table["loan_diff"].abs().sum()
    share = total_diff / table["loans_duckdb"].sum()
    print(f"\nTotal loan-count difference: {total_diff:,} ({share:.4%} of loans)")
    print(f"Max default-rate difference: {table['rate_diff_pp'].abs().max():.3f} pp")


if __name__ == "__main__":
    run_main(main)
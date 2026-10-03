# pylint: disable=no-member
"""Build the loan table with PySpark, mirroring sql/01_build_loan_table.sql."""
import sys
from pyspark.sql import SparkSession, functions as F
from src.credit_default.config import DATA_DIR
from src.credit_default.download import find_accepted_file

COLUMNS = {                       # column -> Spark type
    "id": "string", "loan_amnt": "double", "int_rate": "double",
    "grade": "string", "emp_length": "string", "home_ownership": "string",
    "annual_inc": "double", "dti": "double", "fico_range_low": "double",
    "revol_util": "double", "delinq_2yrs": "double",
    "inq_last_6mths": "double", "open_acc": "double", "pub_rec": "double",
    "purpose": "string", "addr_state": "string",
}


def get_spark() -> SparkSession:
    return (SparkSession.builder
            .appName("credit-default")
            .master("local[*]")                       # use all local cores
            .config("spark.driver.memory", "4g")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def build_loans(spark: SparkSession, raw_path: str):
    raw = (spark.read
           .option("header", True)
           .option("multiLine", True)                 # some text fields span lines
           .option("escape", '"')
           .option("mode", "DROPMALFORMED")
           .csv(raw_path))                            # read as strings: fast, explicit

    loans = (raw
             .filter(F.col("loan_status").isin("Fully Paid", "Charged Off"))
             .filter(F.col("term").contains("36"))
             .withColumn("issue_date", F.to_date("issue_d", "MMM-yyyy"))
             .withColumn("default_flag",
                         (F.col("loan_status") == "Charged Off").cast("int")))

    casts = [F.col(c).cast(t).alias(c) for c, t in COLUMNS.items() if c in loans.columns]
    return loans.select("issue_date", "default_flag", *casts)


def main():
    raw_file = find_accepted_file()
    if raw_file is None:
        sys.exit("No raw data found. Run: python -m src.credit_default.download")

    spark = get_spark()
    loans = build_loans(spark, str(raw_file))

    out = DATA_DIR / "processed" / "loans_spark.parquet"
    loans.write.mode("overwrite").parquet(str(out))

    summary = (spark.read.parquet(str(out))
               .groupBy(F.year("issue_date").alias("year"))
               .agg(F.count("*").alias("loans"),
                    F.round(F.avg("default_flag") * 100, 1).alias("default_rate_pct"))
               .orderBy("year"))
    summary.show(20)
    print(f"Saved {out}")
    spark.stop()


if __name__ == "__main__":
    main()
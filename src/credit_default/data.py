"""Load raw Lending Club data into DuckDB using sql/01_build_loan_table.sql."""
import sys
import duckdb
from src.credit_default.config import ROOT, DATA_DIR
from src.credit_default.download import find_accepted_file


from src.credit_default.logging_utils import get_logger, run_main

log = get_logger(__name__)


def main():
    raw_file = find_accepted_file()
    if raw_file is None:
        sys.exit("No raw data found. Run: python -m src.credit_default.download")

    con = duckdb.connect(str(DATA_DIR / "loans.duckdb"))
    sql = (ROOT / "sql" / "01_build_loan_table.sql").read_text()
    con.execute(sql.replace("{RAW_FILE}", str(raw_file)))

    # Sanity check: loans and default rate by year
    summary = con.execute("""
        SELECT year(issue_date)                  AS year,
               COUNT(*)                          AS loans,
               ROUND(AVG(default_flag) * 100, 1) AS default_rate_pct
        FROM loans
        GROUP BY 1
        ORDER BY 1
    """).df()
    print(summary)

    out = DATA_DIR / "processed" / "loans.parquet"
    con.execute(f"COPY loans TO '{out}' (FORMAT PARQUET)")
    log.info(f"Saved {out}")
if __name__ == "__main__":
    run_main(main)

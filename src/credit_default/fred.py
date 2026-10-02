"""Pull national and state macro data from FRED, load it into DuckDB, and join it to loans."""
import os
import time
import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv
from src.credit_default.config import ROOT, DATA_DIR

load_dotenv()
API_KEY = os.environ["FRED_API_KEY"]
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
START = "2010-01-01"

NATIONAL = {"UNRATE": "unemployment_rate", "FEDFUNDS": "fed_funds_rate"}
STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
          "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
          "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
          "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
          "WV", "WI", "WY"]


def fetch_series(series_id: str) -> pd.DataFrame:
    params = {"series_id": series_id, "api_key": API_KEY,
              "file_type": "json", "observation_start": START}
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["observations"])[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def main():
    con = duckdb.connect(str(DATA_DIR / "loans.duckdb"))

    # National series, merged into one table by date
    national = None
    for sid, name in NATIONAL.items():
        df = fetch_series(sid).rename(columns={"value": name})
        national = df if national is None else national.merge(df, on="date", how="outer")
    con.register("national_df", national)
    con.execute("CREATE OR REPLACE TABLE macro_national AS SELECT * FROM national_df")
    print(f"Loaded national series: {list(NATIONAL)}")

    # State unemployment rates, one series per state (e.g. NCUR, CAUR)
    frames = []
    for st in STATES:
        df = fetch_series(f"{st}UR").rename(columns={"value": "state_unemp"})
        df["state"] = st
        frames.append(df)
        time.sleep(0.5)                       # stay well under FRED's rate limit
    states = pd.concat(frames, ignore_index=True)
    con.register("state_df", states)
    con.execute("CREATE OR REPLACE TABLE macro_state AS SELECT * FROM state_df")
    print(f"Loaded state unemployment for {len(STATES)} states")

    # Join macro data to loans with SQL
    con.execute((ROOT / "sql" / "02_add_macro.sql").read_text())
    missing = con.execute(
        "SELECT AVG(CASE WHEN state_unemp IS NULL THEN 1 ELSE 0 END) FROM loans_macro"
    ).fetchone()[0]
    print(f"Share of loans missing state unemployment: {missing:.2%}")

    out = DATA_DIR / "processed" / "loans_macro.parquet"
    con.execute(f"COPY loans_macro TO '{out}' (FORMAT PARQUET)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

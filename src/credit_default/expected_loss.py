# pylint: disable=wrong-import-position
"""Expected loss: PD x LGD x EAD, validated against realized 2015 losses."""
import sys
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.credit_default.config import ROOT, DATA_DIR, FIG_DIR
from src.credit_default.download import find_accepted_file
from src.credit_default.train import TARGET


def load_loss_outcomes() -> pd.DataFrame:
    """EAD and net recoveries for charged-off loans, read from the raw file."""
    raw_file = find_accepted_file()
    if raw_file is None:
        sys.exit("No raw data found. Run: python -m src.credit_default.download")
    sql = (ROOT / "sql" / "03_loss_outcomes.sql").read_text()
    return duckdb.connect().execute(sql.replace("{RAW_FILE}", str(raw_file))).df()


def loss_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add EAD ratio, LGD, and realized loss; drop loans with no exposure at default."""
    df = df[df["ead"] > 0].copy()
    df["ead_ratio"] = (df["ead"] / df["funded_amnt"]).clip(0, 1)
    df["lgd"] = (1 - df["net_recovery"] / df["ead"]).clip(0, 1)
    df["loss"] = df["ead"] * df["lgd"]
    return df


def expected_loss(pd_: np.ndarray, lgd: float, ead: np.ndarray) -> np.ndarray:
    return pd_ * lgd * ead


def main():
    losses = loss_components(load_loss_outcomes())
    losses["id"] = losses["id"].astype(str)

    train = pd.read_parquet(DATA_DIR / "processed" / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")
    preds = pd.read_parquet(DATA_DIR / "processed" / "test_predictions.parquet")
    for df in (train, test, preds):
        df["id"] = df["id"].astype(str)

    # --- 1. Estimate LGD and EAD from 2012-2014 defaults only ---
    train_def = train[train[TARGET] == 1][["id"]].merge(losses, on="id")
    lgd = train_def["lgd"].mean()
    ead_ratio = train_def["ead_ratio"].mean()
    print(f"\nTraining defaults with loss data: {len(train_def):,}")
    print(f"Estimated LGD: {lgd:.1%}   Estimated EAD ratio: {ead_ratio:.1%}")

    # --- 2. Expected loss on 2015 loans ---
    t = test[["id", TARGET, "loan_amnt"]].merge(
        preds[["id", "pd_logit", "pd_logit_recal"]], on="id")
    t = t.merge(losses[["id", "ead_ratio", "lgd", "loss"]], on="id", how="left")
    t["loss"] = t["loss"].fillna(0.0)                 # non-defaulters lose nothing
    t["ead_hat"] = t["loan_amnt"] * ead_ratio
    t["el_base"] = expected_loss(t["pd_logit"], lgd, t["ead_hat"])
    t["el_recal"] = expected_loss(t["pd_logit_recal"], lgd, t["ead_hat"])

    total_amt = t["loan_amnt"].sum()
    actual_loss = t["loss"].sum()
    portfolio = pd.DataFrame([
        {"estimate": "Base PD", "expected_loss": t["el_base"].sum()},
        {"estimate": "Recalibrated PD", "expected_loss": t["el_recal"].sum()},
        {"estimate": "Actual realized loss", "expected_loss": actual_loss},
    ])
    portfolio["loss_rate"] = portfolio["expected_loss"] / total_amt
    portfolio["vs_actual"] = portfolio["expected_loss"] / actual_loss - 1
    print(f"\n=== 2015 portfolio: {len(t):,} loans, ${total_amt / 1e9:.2f}B originated ===")
    print(portfolio.round(4).to_string(index=False))

    # --- 3. Which component drove the miss? ---
    test_def = t[t[TARGET] == 1].dropna(subset=["lgd"])
    components = pd.DataFrame([
        {"component": "PD (default rate)", "assumed": t["pd_logit"].mean(),
         "actual": t[TARGET].mean()},
        {"component": "LGD", "assumed": lgd, "actual": test_def["lgd"].mean()},
        {"component": "EAD ratio", "assumed": ead_ratio, "actual": test_def["ead_ratio"].mean()},
    ])
    components["ratio_actual_to_assumed"] = components["actual"] / components["assumed"]
    print("\n=== Loss components: assumed (from 2012-2014) vs. actual (2015) ===")
    print(components.round(4).to_string(index=False))

    # --- 4. Expected vs. actual loss rate by PD decile ---
    t["decile"] = pd.qcut(t["pd_logit"], 10, labels=False)
    deciles = t.groupby("decile").agg(
        loans=("id", "size"), amount=("loan_amnt", "sum"),
        expected=("el_base", "sum"), actual=("loss", "sum"))
    deciles["expected_rate"] = deciles["expected"] / deciles["amount"]
    deciles["actual_rate"] = deciles["actual"] / deciles["amount"]
    print("\n=== Loss rate by PD decile (share of originated amount) ===")
    print(deciles[["loans", "expected_rate", "actual_rate"]].round(4).to_string())

    # --- 5. Save outputs ---
    portfolio.to_csv(FIG_DIR / "expected_loss_portfolio.csv", index=False)
    components.to_csv(FIG_DIR / "expected_loss_components.csv", index=False)
    deciles.to_csv(FIG_DIR / "expected_loss_deciles.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(deciles))
    ax.bar(x - 0.2, deciles["expected_rate"], 0.4, label="Expected loss rate")
    ax.bar(x + 0.2, deciles["actual_rate"], 0.4, label="Actual loss rate")
    ax.set_xticks(x); ax.set_xticklabels([str(d + 1) for d in x])
    ax.set_xlabel("PD decile (1 = lowest risk)"); ax.set_ylabel("Loss as share of amount")
    ax.set_title("Expected vs. actual loss by PD decile (2015)"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "expected_loss_deciles.png", dpi=150); plt.close()
    print(f"\nSaved expected loss tables and chart to {FIG_DIR}")


if __name__ == "__main__":
    main()

# pylint: disable=wrong-import-position
"""Monitoring report: feature PSI, score PSI, AUC, and calibration by issue quarter."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.credit_default.config import DATA_DIR, MODEL_DIR, FIG_DIR
from src.credit_default.evaluate import psi
from src.credit_default.train import NUMERIC, CATEGORICAL, TARGET

FEATURES = NUMERIC + CATEGORICAL
PSI_WATCH, PSI_ACT = 0.10, 0.25     # PSI thresholds from the model card
AUC_DROP = 0.03                     # review if AUC falls more than this
CAL_GAP = 0.01                      # review if |actual - predicted| exceeds 1 pp


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """PSI for categorical or low-cardinality features, using category shares."""
    e = expected.astype(str).value_counts(normalize=True)
    a = actual.astype(str).value_counts(normalize=True)
    cats = e.index.union(a.index)
    e = e.reindex(cats, fill_value=0).clip(lower=1e-6).to_numpy()
    a = a.reindex(cats, fill_value=0).clip(lower=1e-6).to_numpy()
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi_value(expected: pd.Series, actual: pd.Series) -> float:
    """Use binned PSI for continuous features, category shares for discrete ones."""
    is_numeric = pd.api.types.is_numeric_dtype(expected)
    if not is_numeric or expected.nunique() <= 10:
        return categorical_psi(expected, actual)
    return psi(expected.to_numpy(), actual.to_numpy())

def psi_status(value: float) -> str:
    if value < PSI_WATCH:
        return "green"
    return "amber" if value < PSI_ACT else "red"


def add_quarter(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(quarter=pd.PeriodIndex(df["issue_date"], freq="Q").astype(str))


def feature_psi_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """PSI of each feature, training vs. all of 2015 and vs. the worst 2015 quarter."""
    rows = []
    for col in FEATURES:
        overall = feature_psi_value(train[col], test[col])
        quarterly = [feature_psi_value(train[col], g[col]) for _, g in test.groupby("quarter")]
        rows.append({"feature": col, "psi_2015": overall, "max_quarter_psi": max(quarterly)})
    table = pd.DataFrame(rows).sort_values("max_quarter_psi", ascending=False)
    table["status"] = table["max_quarter_psi"].map(psi_status)
    return table


def quarterly_table(df: pd.DataFrame, train_scores: np.ndarray, base_auc: float) -> pd.DataFrame:
    """Loans, actual vs. predicted default rate, AUC, and score PSI for each quarter."""
    rows = []
    for (period, quarter), g in df.groupby(["period", "quarter"]):
        auc = roc_auc_score(g[TARGET], g["pd"])
        gap = g[TARGET].mean() - g["pd"].mean()
        score_psi = psi(train_scores, g["pd"].to_numpy())
        flags = []
        if score_psi >= PSI_WATCH:
            flags.append("score PSI")
        if base_auc - auc > AUC_DROP:
            flags.append("AUC drop")
        if abs(gap) > CAL_GAP:
            flags.append("calibration")
        rows.append({"period": period, "quarter": quarter, "loans": len(g),
                     "actual": g[TARGET].mean(), "predicted": g["pd"].mean(),
                     "gap_pp": gap * 100, "auc": auc, "score_psi": score_psi,
                     "flags": ", ".join(flags) or "none"})
    return pd.DataFrame(rows).sort_values("quarter")


def plot_dashboard(q: pd.DataFrame, base_auc: float) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    x = range(len(q))
    test_start = q.index[q["period"] == "test (out-of-time)"].min()
    test_pos = list(q.index).index(test_start)

    axes[0].plot(x, q["actual"], "o-", label="Actual default rate")
    axes[0].plot(x, q["predicted"], "s--", label="Mean predicted PD")
    axes[0].set_ylabel("Default rate"); axes[0].legend()
    axes[0].set_title("Model monitoring by issue quarter")

    axes[1].plot(x, q["auc"], "o-")
    axes[1].axhline(base_auc - AUC_DROP, color="red", linestyle=":", label="Review threshold")
    axes[1].set_ylabel("AUC"); axes[1].legend()

    axes[2].bar(x, q["score_psi"])
    axes[2].axhline(PSI_WATCH, color="orange", linestyle=":", label="Watch (0.10)")
    axes[2].set_ylabel("Score PSI"); axes[2].legend()

    for ax in axes:
        ax.axvspan(test_pos - 0.5, len(q) - 0.5, color="grey", alpha=0.12)
    axes[2].set_xticks(list(x)); axes[2].set_xticklabels(q["quarter"], rotation=45)
    axes[0].text(test_pos, axes[0].get_ylim()[1] * 0.97, "out-of-time", va="top")
    plt.tight_layout(); plt.savefig(FIG_DIR / "monitoring_dashboard.png", dpi=150); plt.close()


def main():
    train = add_quarter(pd.read_parquet(DATA_DIR / "processed" / "train.parquet"))
    test = add_quarter(pd.read_parquet(DATA_DIR / "processed" / "test.parquet"))
    model = joblib.load(MODEL_DIR / "logit.joblib")

    train["pd"] = model.predict_proba(train[FEATURES])[:, 1]
    test["pd"] = model.predict_proba(test[FEATURES])[:, 1]
    train["period"], test["period"] = "train (in-sample)", "test (out-of-time)"
    base_auc = roc_auc_score(train[TARGET], train["pd"])

    feat = feature_psi_table(train, test)
    q = quarterly_table(pd.concat([train, test]), train["pd"].to_numpy(), base_auc)
    q = q.reset_index(drop=True)

    pd.set_option("display.width", 160)
    print(f"\nTraining AUC (baseline for monitoring): {base_auc:.3f}")
    print("\n=== Performance and calibration by issue quarter ===")
    print(q.round(4).to_string(index=False))
    print("\n=== Feature PSI: training vs. 2015 (overall and worst quarter) ===")
    print(feat.round(4).to_string(index=False))

    q.to_csv(FIG_DIR / "monitoring_by_quarter.csv", index=False)
    feat.to_csv(FIG_DIR / "feature_psi.csv", index=False)
    plot_dashboard(q, base_auc)
    print(f"\nSaved monitoring tables and dashboard to {FIG_DIR}")


if __name__ == "__main__":
    main()

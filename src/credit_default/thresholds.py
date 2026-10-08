# pylint: disable=wrong-import-position
"""Threshold analysis: confusion matrices, precision, recall, and approve/decline tradeoffs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.credit_default.config import DATA_DIR, FIG_DIR
from src.credit_default.train import TARGET

THRESHOLDS = np.round(np.arange(0.05, 0.401, 0.025), 3)
MAX_APPROVED_BAD_RATE = 0.10     # example policy: approved loans default at most 10%


def confusion_at(y: np.ndarray, p: np.ndarray, t: float) -> tuple[int, int, int, int]:
    """Counts at threshold t, where PD >= t means 'predict default' (decline)."""
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return tp, fp, tn, fn


def threshold_table(y, p, thresholds=THRESHOLDS) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for t in thresholds:
        tp, fp, tn, fn = confusion_at(y, p, t)
        approved = tn + fn
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
            "approval_rate": approved / len(y),
            "bad_rate_approved": fn / approved if approved else np.nan,
        })
    return pd.DataFrame(rows)


def best_policy(table: pd.DataFrame, max_bad_rate: float = MAX_APPROVED_BAD_RATE):
    """Highest approval rate whose approved-loan default rate stays within the cap."""
    ok = table[table["bad_rate_approved"] <= max_bad_rate]
    return None if ok.empty else ok.loc[ok["approval_rate"].idxmax()]


def main():
    preds = pd.read_parquet(DATA_DIR / "processed" / "test_predictions.parquet")
    y = preds[TARGET].to_numpy()
    table = threshold_table(y, preds["pd_logit"].to_numpy())

    pd.set_option("display.width", 160)
    print(f"\n=== Threshold analysis, logistic regression (2015; default rate {y.mean():.1%}) ===")
    print(table.round(3).to_string(index=False))

    policy = best_policy(table)
    if policy is None:
        print(f"\nNo threshold keeps approved-loan defaults at or below {MAX_APPROVED_BAD_RATE:.0%}.")
    else:
        print(f"\nExample policy: decline PD >= {policy['threshold']:.3f} -> approve "
              f"{policy['approval_rate']:.1%} of applicants with an approved default rate of "
              f"{policy['bad_rate_approved']:.1%} (vs. {y.mean():.1%} if all approved); "
              f"catches {policy['recall']:.1%} of defaulters.")

    table.to_csv(FIG_DIR / "threshold_analysis.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(table["threshold"], table["precision"], "o-", label="Precision")
    axes[0].plot(table["threshold"], table["recall"], "s-", label="Recall")
    axes[0].set_xlabel("PD threshold (decline at or above)"); axes[0].legend()
    axes[0].set_title("Precision and recall by threshold")
    axes[1].plot(table["approval_rate"] * 100, table["bad_rate_approved"] * 100, "o-")
    axes[1].axhline(MAX_APPROVED_BAD_RATE * 100, color="red", linestyle=":", label="Example cap")
    axes[1].set_xlabel("Approval rate (%)"); axes[1].set_ylabel("Default rate of approved loans (%)")
    axes[1].set_title("Approve/decline tradeoff"); axes[1].legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "threshold_analysis.png", dpi=150); plt.close()
    print(f"\nSaved table and chart to {FIG_DIR}")


if __name__ == "__main__":
    main()

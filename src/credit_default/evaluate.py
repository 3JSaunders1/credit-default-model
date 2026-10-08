"""Evaluate models: AUC, Gini, KS, calibration, ROC, and PSI."""
import joblib
import matplotlib
matplotlib.use("Agg")                     # save plots without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from src.credit_default.config import DATA_DIR, MODEL_DIR, FIG_DIR
from src.credit_default.train import NUMERIC, CATEGORICAL, TARGET

MODELS = {"pd_logit": "Logistic regression", "pd_xgb": "XGBoost"}


from src.credit_default.logging_utils import get_logger, run_main

log = get_logger(__name__)


def ks_stat(y, p) -> float:
    """Max gap between the cumulative score distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bucket"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    return (df.groupby("bucket")
              .agg(loans=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
              .round(4))


def psi(expected, actual, bins=10) -> float:
    """Population Stability Index, using deciles of the expected (training) scores."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(DATA_DIR / "processed" / "test_predictions.parquet")
    y = preds[TARGET]

    # --- Metrics table ---
    rows = []
    for col, name in MODELS.items():
        auc = roc_auc_score(y, preds[col])
        rows.append({"model": name, "AUC": auc, "Gini": 2 * auc - 1,
                     "KS": ks_stat(y, preds[col]),
                     "mean_predicted_PD": preds[col].mean(), "actual_rate": y.mean()})
    metrics = pd.DataFrame(rows).round(3)
    print("\n=== Out-of-time test performance (2015 vintage) ===")
    print(metrics.to_string(index=False))

    # --- Calibration by decile (logistic regression) ---
    cal = calibration_table(y, preds["pd_logit"])
    print("\n=== Calibration by risk decile (logistic regression) ===")
    print(cal.to_string())

    plt.figure(figsize=(5, 5))
    plt.plot(cal["predicted"], cal["actual"], "o-", label="Logistic regression")
    lim = max(cal["predicted"].max(), cal["actual"].max()) * 1.05
    plt.plot([0, lim], [0, lim], "--", color="grey", label="Perfect calibration")
    plt.xlabel("Predicted default rate"); plt.ylabel("Actual default rate")
    plt.title("Calibration (2015 out-of-time)"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "calibration.png", dpi=150); plt.close()

    # --- ROC curves ---
    plt.figure(figsize=(5, 5))
    for col, name in MODELS.items():
        fpr, tpr, _ = roc_curve(y, preds[col])
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, preds[col]):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve (2015 out-of-time)"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "roc.png", dpi=150); plt.close()

    # --- PSI: training scores vs. test scores ---
    train = pd.read_parquet(DATA_DIR / "processed" / "train.parquet")
    logit = joblib.load(MODEL_DIR / "logit.joblib")
    train_scores = logit.predict_proba(train[NUMERIC + CATEGORICAL])[:, 1]
    score_psi = psi(train_scores, preds["pd_logit"])
    print(f"\n=== Score PSI, train (2012–2014) vs. test (2015): {score_psi:.3f} ===")

    metrics.to_csv(FIG_DIR / "metrics.csv", index=False)
    cal.to_csv(FIG_DIR / "calibration_table.csv")
    log.info(f"Saved figures and tables to {FIG_DIR}")
if __name__ == "__main__":
    run_main(main)
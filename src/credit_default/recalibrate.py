# pylint: disable=wrong-import-position,no-name-in-module
"""Recalibrate the logistic model's baseline using the latest observed vintage (2014)."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from src.credit_default.config import DATA_DIR, MODEL_DIR, FIG_DIR
from src.credit_default.train import NUMERIC, CATEGORICAL, TARGET
from src.credit_default.evaluate import calibration_table

RECAL_START = "2014-01-01"   # most recent vintage fully observed before 2015


def intercept_shift(p: np.ndarray, target_rate: float) -> float:
    """Find the log-odds shift b so the average adjusted PD equals target_rate."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return brentq(lambda b: expit(z + b).mean() - target_rate, -5, 5)


def apply_shift(p: np.ndarray, b: float) -> np.ndarray:
    return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) + b)


def main():
    train = pd.read_parquet(DATA_DIR / "processed" / "train.parquet")
    model = joblib.load(MODEL_DIR / "logit.joblib")

    # 1. Measure the shift on the 2014 vintage only
    recal = train[train["issue_date"] >= RECAL_START]
    p_recal = model.predict_proba(recal[NUMERIC + CATEGORICAL])[:, 1]
    target = recal[TARGET].mean()
    b = intercept_shift(p_recal, target)
    print(f"2014 vintage: predicted {p_recal.mean():.2%}, actual {target:.2%}")
    print(f"Intercept shift (log-odds): {b:+.3f}")

    # 2. Apply it to the 2015 out-of-time predictions
    preds = pd.read_parquet(DATA_DIR / "processed" / "test_predictions.parquet")
    preds["pd_logit_recal"] = apply_shift(preds["pd_logit"].to_numpy(), b)
    y = preds[TARGET]

    print("\n=== 2015 out-of-time calibration ===")
    print(f"Actual default rate:        {y.mean():.2%}")
    print(f"Mean PD before recalibration: {preds['pd_logit'].mean():.2%}")
    print(f"Mean PD after recalibration:  {preds['pd_logit_recal'].mean():.2%}")

    before = calibration_table(y, preds["pd_logit"])
    after = calibration_table(y, preds["pd_logit_recal"])

    # 3. Before-and-after calibration plot
    plt.figure(figsize=(5, 5))
    plt.plot(before["predicted"], before["actual"], "o-", label="Before")
    plt.plot(after["predicted"], after["actual"], "s-", label="After (2014 recalibration)")
    lim = max(before["actual"].max(), after["predicted"].max()) * 1.05
    plt.plot([0, lim], [0, lim], "--", color="grey", label="Perfect calibration")
    plt.xlabel("Predicted default rate"); plt.ylabel("Actual default rate")
    plt.title("Calibration before and after recalibration"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "calibration_recalibrated.png", dpi=150); plt.close()

    joblib.dump({"intercept_shift": b, "fit_on": "2014 vintage"},
                MODEL_DIR / "recalibration.joblib")
    preds.to_parquet(DATA_DIR / "processed" / "test_predictions.parquet")
    after.to_csv(FIG_DIR / "calibration_table_recalibrated.csv")
    print(f"\nSaved plot and recalibration to {FIG_DIR} and {MODEL_DIR}")


if __name__ == "__main__":
    main()
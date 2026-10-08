# pylint: disable=wrong-import-position,no-name-in-module
"""Recalibrate the logistic model's baseline, comparing an original and a strict design."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from src.credit_default.config import DATA_DIR, MODEL_DIR, FIG_DIR
from src.credit_default.train import NUMERIC, CATEGORICAL, TARGET, make_logit
from src.credit_default.evaluate import calibration_table

RECAL_START = "2014-01-01"   # most recent vintage fully observed before 2015
FEATURES = NUMERIC + CATEGORICAL


from src.credit_default.logging_utils import get_logger, run_main

log = get_logger(__name__)


def intercept_shift(p: np.ndarray, target_rate: float) -> float:
    """Find the log-odds shift b so the average adjusted PD equals target_rate."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return brentq(lambda b: expit(z + b).mean() - target_rate, -5, 5)


def apply_shift(p: np.ndarray, b: float) -> np.ndarray:
    return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) + b)


def split_for_recalibration(train: pd.DataFrame):
    """Split training data into a fit period (2012-2013) and a recalibration holdout (2014)."""
    fit = train[train["issue_date"] < RECAL_START]
    holdout = train[train["issue_date"] >= RECAL_START]
    return fit, holdout


def original_recalibration(train: pd.DataFrame, preds: pd.DataFrame) -> dict:
    """Original design: the production model (trained 2012-2014), recalibrated on 2014."""
    model = joblib.load(MODEL_DIR / "logit.joblib")
    _, holdout = split_for_recalibration(train)
    p_hold = model.predict_proba(holdout[FEATURES])[:, 1]
    b = intercept_shift(p_hold, holdout[TARGET].mean())
    p_test = preds["pd_logit"].to_numpy()
    return {"design": "Original (2014 in training)",
            "holdout_predicted": p_hold.mean(), "holdout_actual": holdout[TARGET].mean(),
            "shift": b, "test_before": p_test.mean(),
            "test_after": apply_shift(p_test, b).mean(),
            "test_auc": roc_auc_score(preds[TARGET], p_test),
            "p_after": apply_shift(p_test, b)}


def strict_recalibration(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Strict design: train on 2012-2013, recalibrate on a true 2014 holdout, apply to 2015."""
    fit, holdout = split_for_recalibration(train)
    model = make_logit().fit(fit[FEATURES], fit[TARGET])
    p_hold = model.predict_proba(holdout[FEATURES])[:, 1]
    b = intercept_shift(p_hold, holdout[TARGET].mean())
    p_test = model.predict_proba(test[FEATURES])[:, 1]
    return {"design": "Strict (2014 held out)",
            "holdout_predicted": p_hold.mean(), "holdout_actual": holdout[TARGET].mean(),
            "shift": b, "test_before": p_test.mean(),
            "test_after": apply_shift(p_test, b).mean(),
            "test_auc": roc_auc_score(test[TARGET], p_test),
            "p_after": apply_shift(p_test, b), "model": model}


def main():
    train = pd.read_parquet(DATA_DIR / "processed" / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")
    preds = pd.read_parquet(DATA_DIR / "processed" / "test_predictions.parquet")
    y = preds[TARGET]

    original = original_recalibration(train, preds)
    strict = strict_recalibration(train, test)

    # --- Comparison table ---
    cols = ["design", "holdout_predicted", "holdout_actual", "shift",
            "test_before", "test_after", "test_auc"]
    comparison = pd.DataFrame([{k: r[k] for k in cols} for r in (original, strict)])
    print(f"\n=== Recalibration designs compared (2015 actual default rate: {y.mean():.2%}) ===")
    print(comparison.round(4).to_string(index=False))

    # --- Calibration plot: before, original, and strict ---
    before = calibration_table(y, preds["pd_logit"])
    after_orig = calibration_table(y, original["p_after"])
    after_strict = calibration_table(test[TARGET], strict["p_after"])

    plt.figure(figsize=(5.5, 5.5))
    plt.plot(before["predicted"], before["actual"], "o-", label="Before recalibration")
    plt.plot(after_orig["predicted"], after_orig["actual"], "s-", label="Original design")
    plt.plot(after_strict["predicted"], after_strict["actual"], "^-", label="Strict design")
    lim = max(before["actual"].max(), after_strict["predicted"].max()) * 1.05
    plt.plot([0, lim], [0, lim], "--", color="grey", label="Perfect calibration")
    plt.xlabel("Predicted default rate"); plt.ylabel("Actual default rate")
    plt.title("Calibration: recalibration designs (2015)"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "calibration_recalibrated.png", dpi=150); plt.close()

    # --- Save outputs ---
    preds["pd_logit_recal"] = original["p_after"]
    preds.to_parquet(DATA_DIR / "processed" / "test_predictions.parquet")
    joblib.dump({"original_shift": original["shift"], "strict_shift": strict["shift"],
                 "strict_model": strict["model"], "holdout": "2014 vintage"},
                MODEL_DIR / "recalibration.joblib")
    comparison.to_csv(FIG_DIR / "recalibration_comparison.csv", index=False)
    after_strict.to_csv(FIG_DIR / "calibration_table_recalibrated.csv")
    log.info(f"Saved comparison, plot, and recalibration to {FIG_DIR} and {MODEL_DIR}")
if __name__ == "__main__":
    run_main(main)
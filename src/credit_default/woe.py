# pylint: disable=wrong-import-position
"""Weight of Evidence (WoE) scorecard: binning, information value, logistic fit, and points."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.credit_default.config import DATA_DIR, FIG_DIR, MODEL_DIR
from src.credit_default.evaluate import ks_stat
from src.credit_default.train import NUMERIC, CATEGORICAL, TARGET

N_BINS = 10            # quantile bins for continuous features
MIN_IV = 0.02          # drop features below this information value ("weak")
RARE_SHARE = 0.01      # categories under 1% are grouped
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20   # 600 points = 50:1 odds; +20 points doubles odds
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)
RARE = "RARE/UNSEEN"


from src.credit_default.logging_utils import get_logger, run_main

log = get_logger(__name__)


def is_categorical(s: pd.Series) -> bool:
    return (not pd.api.types.is_numeric_dtype(s)) or s.nunique() <= 10


class WoEBinner:
    """Fit bins and WoE values on training data, then apply them to new data."""

    def __init__(self, n_bins: int = N_BINS):
        self.n_bins = n_bins
        self.bins_, self.woe_, self.iv_ = {}, {}, {}

    def _labels(self, col: str, s: pd.Series) -> pd.Series:
        kind, spec = self.bins_[col]
        if kind == "num":
            return pd.cut(s, spec, include_lowest=True).astype(str)
        s = s.astype(str)
        return s.where(s.isin(spec), RARE)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for col in X.columns:
            s = X[col]
            if is_categorical(s):
                shares = s.astype(str).value_counts(normalize=True)
                self.bins_[col] = ("cat", list(shares[shares >= RARE_SHARE].index))
            else:
                edges = np.unique(np.quantile(s, np.linspace(0, 1, self.n_bins + 1)))
                edges[0], edges[-1] = -np.inf, np.inf
                self.bins_[col] = ("num", edges)

            table = pd.crosstab(self._labels(col, s), y) + 0.5   # smoothing avoids log(0)
            dist_good = table[0] / table[0].sum()
            dist_bad = table[1] / table[1].sum()
            woe = np.log(dist_good / dist_bad)
            self.woe_[col] = woe
            self.iv_[col] = float(((dist_good - dist_bad) * woe).sum())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            out[col] = self._labels(col, X[col]).map(self.woe_[col]).fillna(0.0).astype(float)
        return out


def score_from_pd(p: np.ndarray) -> np.ndarray:
    """Convert PD to scorecard points: OFFSET + FACTOR * ln(odds of not defaulting)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return OFFSET + FACTOR * np.log((1 - p) / p)


def build_scorecard(binner: WoEBinner, model: LogisticRegression, features: list[str]) -> pd.DataFrame:
    """Points for every bin of every feature; a loan's score is the sum of its bins' points."""
    b0, n = model.intercept_[0], len(features)
    rows = []
    for col, beta in zip(features, model.coef_[0]):
        for label, w in binner.woe_[col].items():
            points = -FACTOR * beta * w + (OFFSET - FACTOR * b0) / n
            rows.append({"feature": col, "bin": label, "woe": round(w, 4), "points": round(points)})
    return pd.DataFrame(rows)


def main():
    train = pd.read_parquet(DATA_DIR / "processed" / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")
    features = NUMERIC + CATEGORICAL

    # 1. Bin and compute WoE and IV on training data only
    binner = WoEBinner().fit(train[features], train[TARGET])
    iv = (pd.Series(binner.iv_).sort_values(ascending=False)
            .rename("iv").rename_axis("feature").reset_index())
    selected = iv.loc[iv["iv"] >= MIN_IV, "feature"].tolist()
    print("\n=== Information value (training data) ===")
    print(iv.round(4).to_string(index=False))
    print(f"\nSelected {len(selected)} of {len(features)} features with IV >= {MIN_IV}")

    # 2. Logistic regression on WoE values
    X_train = binner.transform(train[selected])
    X_test = binner.transform(test[selected])
    model = LogisticRegression(max_iter=1000).fit(X_train, train[TARGET])
    coefs = pd.Series(model.coef_[0], index=selected)
    wrong_sign = coefs[coefs > 0].index.tolist()
    print("\nCoefficient signs:", "all negative, as expected" if not wrong_sign
          else f"unexpected positive sign for {wrong_sign}")

    # 3. Out-of-time evaluation vs. the benchmark logistic regression
    y = test[TARGET]
    p = model.predict_proba(X_test)[:, 1]
    bench = pd.read_parquet(DATA_DIR / "processed" / "test_predictions.parquet")["pd_logit"]
    results = pd.DataFrame([
        {"model": "Benchmark logistic (standardized)", "AUC": roc_auc_score(y, bench),
         "KS": ks_stat(y, bench), "mean_PD": bench.mean()},
        {"model": "WoE scorecard", "AUC": roc_auc_score(y, p),
         "KS": ks_stat(y, p), "mean_PD": p.mean()},
    ])
    results["Gini"] = 2 * results["AUC"] - 1
    print(f"\n=== 2015 out-of-time performance (actual default rate {y.mean():.2%}) ===")
    print(results.round(4).to_string(index=False))

    # 4. Scorecard points and score bands
    scorecard = build_scorecard(binner, model, selected)
    scores = score_from_pd(p)
    bands = pd.qcut(scores, 10, duplicates="drop")
    band_table = (pd.DataFrame({"band": bands, "default": y.to_numpy(), "pd": p})
                    .groupby("band", observed=True)
                    .agg(loans=("default", "size"), actual=("default", "mean"),
                         predicted=("pd", "mean")))
    print("\n=== Default rate by score band (2015) ===")
    print(band_table.round(4).to_string())
    print(f"\nScore range: {scores.min():.0f} to {scores.max():.0f}, median {np.median(scores):.0f}")

    # 5. Save outputs
    iv.to_csv(FIG_DIR / "woe_information_value.csv", index=False)
    scorecard.to_csv(FIG_DIR / "woe_scorecard_points.csv", index=False)
    band_table.to_csv(FIG_DIR / "woe_score_bands.csv")
    joblib.dump({"binner": binner, "model": model, "features": selected},
                MODEL_DIR / "woe_scorecard.joblib")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{int(i.left)}–{int(i.right)}" for i in band_table.index]
    ax.bar(labels, band_table["actual"], label="Actual default rate")
    ax.plot(labels, band_table["predicted"], "ko--", label="Predicted PD")
    ax.set_xlabel("Score band (higher = safer)"); ax.set_ylabel("Default rate")
    ax.set_title("WoE scorecard: default rate by score band (2015)"); ax.legend()
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(FIG_DIR / "woe_score_bands.png", dpi=150); plt.close()
    log.info(f"Saved scorecard, IV table, score bands, and chart to {FIG_DIR}")
if __name__ == "__main__":
    run_main(main)

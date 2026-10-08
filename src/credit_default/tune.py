"""Hyperparameter tuning with time-aware (expanding-window) cross-validation."""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.credit_default.config import DATA_DIR, MODEL_DIR, FIG_DIR, RANDOM_SEED
from src.credit_default.train import NUMERIC, CATEGORICAL, TARGET, make_preprocessor

FEATURES = NUMERIC + CATEGORICAL
N_ITER = 20
# Each fold trains on everything before `start` and validates on [start, end)
FOLDS = [("2013-07-01", "2014-01-01"),
         ("2014-01-01", "2014-07-01"),
         ("2014-07-01", "2015-01-01")]
BASELINE = {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0}


from src.credit_default.logging_utils import get_logger, run_main

log = get_logger(__name__)


def time_folds(df: pd.DataFrame):
    """Expanding-window folds: always train on the past, validate on the future."""
    for start, end in FOLDS:
        train = df[df["issue_date"] < start]
        valid = df[(df["issue_date"] >= start) & (df["issue_date"] < end)]
        yield train, valid


def sample_params(rng: np.random.Generator) -> dict:
    return {"max_depth": int(rng.integers(2, 7)),
            "learning_rate": float(rng.choice([0.02, 0.05, 0.1])),
            "min_child_weight": int(rng.choice([1, 10, 50, 100])),
            "subsample": float(rng.choice([0.6, 0.8, 1.0])),
            "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
            "reg_lambda": float(rng.choice([0.5, 1.0, 5.0, 10.0]))}


def cv_xgb(df: pd.DataFrame, params: dict) -> tuple[float, float, int]:
    """Mean and std of validation AUC across folds, plus the average best tree count."""
    aucs, trees = [], []
    for train, valid in time_folds(df):
        prep = make_preprocessor().fit(train[FEATURES])
        X_tr, X_va = prep.transform(train[FEATURES]), prep.transform(valid[FEATURES])
        model = XGBClassifier(n_estimators=1000, eval_metric="auc", early_stopping_rounds=50,
                              random_state=RANDOM_SEED, n_jobs=-1, **params)
        model.fit(X_tr, train[TARGET], eval_set=[(X_va, valid[TARGET])], verbose=False)
        aucs.append(roc_auc_score(valid[TARGET], model.predict_proba(X_va)[:, 1]))
        trees.append(model.best_iteration + 1)
    return float(np.mean(aucs)), float(np.std(aucs)), int(np.mean(trees))


def cv_logit(df: pd.DataFrame, c: float) -> tuple[float, float]:
    aucs = []
    for train, valid in time_folds(df):
        model = Pipeline([("prep", make_preprocessor()),
                          ("model", LogisticRegression(C=c, max_iter=1000))])
        model.fit(train[FEATURES], train[TARGET])
        aucs.append(roc_auc_score(valid[TARGET], model.predict_proba(valid[FEATURES])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    train = pd.read_parquet(DATA_DIR / "processed" / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")
    rng = np.random.default_rng(RANDOM_SEED)

    # --- XGBoost random search (baseline settings included as candidate 0) ---
    candidates = [BASELINE] + [sample_params(rng) for _ in range(N_ITER)]
    rows = []
    for i, params in enumerate(candidates):
        mean_auc, std_auc, n_trees = cv_xgb(train, params)
        rows.append({"candidate": i, **params, "cv_auc": mean_auc,
                     "cv_std": std_auc, "trees": n_trees})
        print(f"[{i:2d}/{len(candidates) - 1}] CV AUC {mean_auc:.4f} ± {std_auc:.4f}  {params}")
    xgb_results = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)

    # --- Logistic regression: regularization strength ---
    logit_rows = []
    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        mean_auc, std_auc = cv_logit(train, c)
        logit_rows.append({"C": c, "cv_auc": mean_auc, "cv_std": std_auc})
    logit_results = pd.DataFrame(logit_rows).sort_values("cv_auc", ascending=False)

    # --- Refit the best settings on all of 2012-2014, evaluate on 2015 once ---
    best = xgb_results.iloc[0]
    best_params = {k: best[k] for k in BASELINE}
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_child_weight"] = int(best_params["min_child_weight"])
    prep = make_preprocessor().fit(train[FEATURES])
    xgb = XGBClassifier(n_estimators=int(best["trees"]), random_state=RANDOM_SEED,
                        n_jobs=-1, **best_params)
    xgb.fit(prep.transform(train[FEATURES]), train[TARGET])
    y = test[TARGET]
    xgb_test_auc = roc_auc_score(y, xgb.predict_proba(prep.transform(test[FEATURES]))[:, 1])

    best_c = float(logit_results.iloc[0]["C"])
    logit = Pipeline([("prep", make_preprocessor()),
                      ("model", LogisticRegression(C=best_c, max_iter=1000))])
    logit.fit(train[FEATURES], train[TARGET])
    logit_test_auc = roc_auc_score(y, logit.predict_proba(test[FEATURES])[:, 1])

    preds = pd.read_parquet(DATA_DIR / "processed" / "test_predictions.parquet")
    summary = pd.DataFrame([
        {"model": "XGBoost (default settings)", "test_auc_2015": roc_auc_score(y, preds["pd_xgb"])},
        {"model": "XGBoost (tuned)", "test_auc_2015": xgb_test_auc},
        {"model": "Logistic (C=1.0, default)", "test_auc_2015": roc_auc_score(y, preds["pd_logit"])},
        {"model": f"Logistic (tuned, C={best_c})", "test_auc_2015": logit_test_auc},
    ])

    pd.set_option("display.width", 180)
    print("\n=== Top 5 XGBoost configurations (time-aware CV) ===")
    print(xgb_results.head(5).round(4).to_string(index=False))
    print(f"\nBaseline settings ranked #{list(xgb_results['candidate']).index(0) + 1} "
          f"of {len(candidates)}")
    print("\n=== Logistic regression regularization (time-aware CV) ===")
    print(logit_results.round(4).to_string(index=False))
    print("\n=== 2015 out-of-time AUC: default vs. tuned ===")
    print(summary.round(4).to_string(index=False))

    xgb_results.to_csv(FIG_DIR / "tuning_xgb.csv", index=False)
    logit_results.to_csv(FIG_DIR / "tuning_logit.csv", index=False)
    summary.to_csv(FIG_DIR / "tuning_summary.csv", index=False)
    joblib.dump({"prep": prep, "model": xgb, "params": best_params}, MODEL_DIR / "xgb_tuned.joblib")
    log.info(f"Saved tuning results to {FIG_DIR}")
if __name__ == "__main__":
    run_main(main)

"""Explainability: odds ratios, SHAP, partial dependence, and monotonic constraints."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.credit_default.config import DATA_DIR, MODEL_DIR, FIG_DIR, RANDOM_SEED
from src.credit_default.train import NUMERIC, CATEGORICAL, TARGET

FEATURES = NUMERIC + CATEGORICAL
SAMPLE_SIZE = 5000

# Expected direction of each feature's effect on default risk
# (+1 = can only increase risk, -1 = can only decrease, 0 = unconstrained)
DIRECTIONS = {
    "num__dti": 1, "num__revol_util": 1, "num__inq_last_6mths": 1,
    "num__delinq_2yrs": 1, "num__pub_rec": 1, "num__loan_to_income": 1,
    "num__fico_range_low": -1, "num__log_annual_inc": -1,
}


def dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def odds_ratios(logit_pipe):
    """Compare: interpretable logistic coefficients as odds ratios."""
    names = logit_pipe.named_steps["prep"].get_feature_names_out()
    coefs = logit_pipe.named_steps["model"].coef_[0]
    table = (pd.DataFrame({"feature": names, "coef": coefs,
                           "odds_ratio": np.exp(coefs)})
               .assign(abs_coef=lambda d: d["coef"].abs())
               .sort_values("abs_coef", ascending=False)
               .drop(columns="abs_coef"))
    table.to_csv(FIG_DIR / "logit_odds_ratios.csv", index=False)
    print("\n=== Top logistic regression effects (odds ratio per 1 SD for numeric) ===")
    print(table.head(10).round(3).to_string(index=False))


def shap_analysis(prep, model, sample):
    """Show: SHAP values, computed natively by XGBoost."""
    names = list(prep.get_feature_names_out())
    X = dense(prep.transform(sample[FEATURES]))
    contribs = model.get_booster().predict(
        xgb.DMatrix(X, feature_names=names), pred_contribs=True)
    shap_values = contribs[:, :-1]               # last column is the baseline

    shap.summary_plot(shap_values, X, feature_names=names, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(FIG_DIR / "shap_summary.png", dpi=150); plt.close()

    shap.summary_plot(shap_values, X, feature_names=names, plot_type="bar",
                      show=False, max_display=12)
    plt.tight_layout(); plt.savefig(FIG_DIR / "shap_importance.png", dpi=150); plt.close()

    # Reason codes for the riskiest loan in the sample
    i = int(np.argmax(model.predict_proba(X)[:, 1]))
    top = pd.Series(shap_values[i], index=names).sort_values(ascending=False).head(5)
    print("\n=== Top 5 reasons for the highest-risk loan in the sample ===")
    print(top.round(3).to_string())


def partial_dependence(prep, model, sample):
    """Plot: how predicted risk changes as one feature changes."""
    pipe = Pipeline([("prep", prep), ("model", model)])
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        pipe, sample[FEATURES],
        features=["dti", "fico_range_low", "revol_util", "loan_to_income"],
        ax=ax)
    plt.tight_layout(); plt.savefig(FIG_DIR / "partial_dependence.png", dpi=150); plt.close()


def monotonic_model(prep, train, test, base_auc):
    """Constrain: retrain XGBoost with monotonic constraints and compare AUC."""
    names = prep.get_feature_names_out()
    constraints = tuple(DIRECTIONS.get(n, 0) for n in names)

    fit_part = train["issue_date"] < "2014-01-01"
    X_fit = dense(prep.transform(train.loc[fit_part, FEATURES]))
    X_val = dense(prep.transform(train.loc[~fit_part, FEATURES]))
    X_test = dense(prep.transform(test[FEATURES]))

    mono = XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", early_stopping_rounds=50,
        monotone_constraints=constraints, random_state=RANDOM_SEED)
    mono.fit(X_fit, train.loc[fit_part, TARGET],
             eval_set=[(X_val, train.loc[~fit_part, TARGET])], verbose=False)

    auc = roc_auc_score(test[TARGET], mono.predict_proba(X_test)[:, 1])
    print(f"\n=== Monotonic constraints ===")
    print(f"Unconstrained XGBoost AUC: {base_auc:.3f}")
    print(f"Monotonic XGBoost AUC:     {auc:.3f}")
    joblib.dump({"prep": prep, "model": mono}, MODEL_DIR / "xgb_monotonic.joblib")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DATA_DIR / "processed" / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")
    logit_pipe = joblib.load(MODEL_DIR / "logit.joblib")
    bundle = joblib.load(MODEL_DIR / "xgb.joblib")
    prep, model = bundle["prep"], bundle["model"]

    sample = test.sample(SAMPLE_SIZE, random_state=RANDOM_SEED)
    base_auc = roc_auc_score(
        test[TARGET], model.predict_proba(dense(prep.transform(test[FEATURES])))[:, 1])

    odds_ratios(logit_pipe)
    shap_analysis(prep, model, sample)
    partial_dependence(prep, model, sample)
    monotonic_model(prep, train, test, base_auc)
    print(f"\nSaved explainability figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
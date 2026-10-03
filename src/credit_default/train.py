"""Train a logistic regression benchmark and an XGBoost challenger."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from src.credit_default.config import DATA_DIR, MODEL_DIR, RANDOM_SEED

NUMERIC = ["loan_amnt", "log_annual_inc", "dti", "fico_range_low", "revol_util",
           "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
           "emp_length_yrs", "loan_to_income",
           "emp_length_yrs_missing", "dti_missing", "revol_util_missing"]
CATEGORICAL = ["home_ownership", "purpose"]
TARGET = "default_flag"


def make_preprocessor(numeric: list[str] | None = None,
                      categorical: list[str] | None = None) -> ColumnTransformer:
    """Single source of truth for preprocessing, shared by every script."""
    numeric = NUMERIC if numeric is None else numeric
    categorical = CATEGORICAL if categorical is None else categorical
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=0.01, drop="first"), categorical),
    ])


def make_logit(numeric: list[str] | None = None,
               categorical: list[str] | None = None) -> Pipeline:
    """The benchmark logistic regression, with shared preprocessing."""
    return Pipeline([
        ("prep", make_preprocessor(numeric, categorical)),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def main():
    train = pd.read_parquet(DATA_DIR / "processed" / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "processed" / "test.parquet")
    X_train, y_train = train[NUMERIC + CATEGORICAL], train[TARGET]
    X_test, y_test = test[NUMERIC + CATEGORICAL], test[TARGET]

    # --- Benchmark: logistic regression ---
    logit = make_logit()
    logit.fit(X_train, y_train)

    # --- Challenger: XGBoost with time-aware early stopping ---
    fit_part = train["issue_date"] < "2014-01-01"     # 2012–2013
    prep = make_preprocessor().fit(X_train[fit_part])
    X_fit = prep.transform(X_train[fit_part])
    X_val = prep.transform(X_train[~fit_part])        # 2014 vintage

    xgb = XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", early_stopping_rounds=50,
        random_state=RANDOM_SEED,
    )
    xgb.fit(X_fit, y_train[fit_part],
            eval_set=[(X_val, y_train[~fit_part])], verbose=False)
    print(f"XGBoost stopped at {xgb.best_iteration} trees")

    # --- Save models and test predictions ---
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(logit, MODEL_DIR / "logit.joblib")
    joblib.dump({"prep": prep, "model": xgb}, MODEL_DIR / "xgb.joblib")

    preds = test[["id", "issue_date", TARGET]].copy()
    preds["pd_logit"] = logit.predict_proba(X_test)[:, 1]
    preds["pd_xgb"] = xgb.predict_proba(prep.transform(X_test))[:, 1]
    preds.to_parquet(DATA_DIR / "processed" / "test_predictions.parquet")

    for name in ["pd_logit", "pd_xgb"]:
        auc = roc_auc_score(preds[TARGET], preds[name])
        print(f"{name}: test AUC = {auc:.3f}, Gini = {2 * auc - 1:.3f}")


if __name__ == "__main__":
    main()
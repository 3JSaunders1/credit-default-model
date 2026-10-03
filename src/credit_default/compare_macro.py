"""Compare logistic regression with and without macro features on the 2015 out-of-time test."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.credit_default.config import DATA_DIR, FIG_DIR
from src.credit_default.features import clean, split
from src.credit_default.train import NUMERIC, CATEGORICAL, TARGET, make_logit
from src.credit_default.evaluate import calibration_table

MACRO = ["unemployment_rate", "fed_funds_rate", "state_unemp", "state_unemp_chg_12m"]


def main():
    df = clean(pd.read_parquet(DATA_DIR / "processed" / "loans_macro.parquet"))
    train, test = split(df)

    fill = NUMERIC + MACRO
    medians = train[fill].median()             # training medians only
    train, test = train.fillna(medians), test.fillna(medians)
    y = test[TARGET]

    results, preds, models = [], {}, {}
    for name, numeric in [("Base", NUMERIC), ("Base + macro", NUMERIC + MACRO)]:
        model = make_logit(numeric).fit(train[numeric + CATEGORICAL], train[TARGET])
        p = model.predict_proba(test[numeric + CATEGORICAL])[:, 1]
        models[name], preds[name] = model, p
        auc = roc_auc_score(y, p)
        results.append({"model": name, "AUC": round(auc, 3), "Gini": round(2 * auc - 1, 3),
                        "mean_PD": f"{p.mean():.2%}", "actual": f"{y.mean():.2%}"})

    # Macro odds ratios from the macro-enhanced model
    macro_model = models["Base + macro"]
    names = macro_model.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(macro_model.named_steps["model"].coef_[0], index=names)
    macro_or = np.exp(coefs[[f"num__{m}" for m in MACRO]]).round(3)

    print("\n=== Base vs. macro-enhanced model (2015 out-of-time) ===")
    print(pd.DataFrame(results).to_string(index=False))

    print("\n=== Macro odds ratios (per 1 SD) ===")
    print(macro_or.to_string())

    cal = pd.DataFrame({
        "base_pred": calibration_table(y, preds["Base"])["predicted"],
        "macro_pred": calibration_table(y, preds["Base + macro"])["predicted"],
        "actual": calibration_table(y, preds["Base"])["actual"],
    })
    print("\n=== Calibration by decile ===")
    print(cal.round(4).to_string())

    pd.DataFrame(results).to_csv(FIG_DIR / "macro_comparison.csv", index=False)


if __name__ == "__main__":
    main()

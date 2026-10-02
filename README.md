# Credit Default Prediction Model

[![Tests](https://github.com/3JSaunders1/credit-default-model/actions/workflows/tests.yml/badge.svg)](https://github.com/3JSaunders1/credit-default-model/actions/workflows/tests.yml)

An end-to-end probability of default (PD) model built on Lending Club loan data, developed and evaluated the way a bank would evaluate a credit model before relying on it: out-of-time validation, discrimination and calibration metrics, population stability monitoring, recalibration, macroeconomic features from the FRED API, explainability, automated tests, and an honest account of limitations.

---

## Key Findings at a Glance

| | Result |
|---|---|
| **Out-of-time AUC (2015 vintage)** | 0.655 (Gini 0.31, KS 0.22) |
| **Risk separation** | Actual default rates rise from **4.3%** in the lowest-risk decile to **28.1%** in the highest, about **6.6x** |
| **Logistic regression vs. XGBoost** | Virtually identical performance, so the interpretable model is preferred |
| **Calibration** | The base model **underpredicts** 2015 defaults: 13.2% predicted vs. 14.9% actual |
| **Population stability** | Score PSI of **0.001**: no meaningful shift in the borrower population |
| **Main insight** | The 2015 miss is **concept drift, not data drift**. Borrowers looked the same, but defaulted more. PSI alone would not have caught it. |
| **Recalibration** | Recalibrating on the latest available vintage (2014) moved the mean PD only from 13.2% to 13.4% |
| **Macro features** | Cut the calibration gap roughly in half (14.1% vs. 14.9% actual), but national unemployment showed a counterintuitive, cycle-driven sign that should not be trusted in production |
| **Explainability** | Drivers match credit intuition (loan purpose, FICO, renting, inquiries, loan-to-income). Monotonic constraints cost essentially **no** AUC (0.654 vs. 0.655). |
| **Engineering** | 21 automated tests run on every push, pinned dependencies, a one-command pipeline, and timestamped run archives |

---

## 1. Purpose

The goal is to predict the probability that a 36-month consumer loan is **charged off at any point over its term**, using only information available **at origination**, and to evaluate the model with the rigor expected in bank model risk management.

The project emphasizes:

- **Avoiding leakage**: only features known when the loan was issued.
- **Out-of-time validation**: training on earlier vintages and testing on a later one, which mirrors how a credit model is actually used.
- **Calibration, not just ranking**: PDs feed expected loss (PD × LGD × EAD), reserves, pricing, and capital, so the probability levels need to be right, not just the ordering.
- **Monitoring**: checking population stability to understand whether performance changes come from data drift or concept drift.
- **Explainability**: making model behavior transparent and defensible, as lenders must be able to explain credit decisions.
- **Questioning results**: testing whether an improvement is trustworthy, not just whether a metric went up.
- **Reproducibility**: tests, pinned dependencies, and archived runs, so every result can be traced and rerun.

This complements my [Macro-Driven Credit Risk Lab](https://github.com/3JSaunders1/macro-credit-risk-lab), which models how macroeconomic conditions drive credit risk. This project focuses on borrower-level default prediction.

---

## 2. Data

### Loan data
- **Lending Club accepted loans, 2007–2018 Q4**, from the public Kaggle dataset `wordsforthewise/lending-club`.
- Downloaded programmatically through the Kaggle API (`src/credit_default/download.py`). Only the accepted-loans file is downloaded, and it is read directly in compressed form to save disk space.

### Macroeconomic data
- **FRED API** (Federal Reserve Bank of St. Louis): national unemployment rate, federal funds rate, and state unemployment rates for all 50 states plus DC (`src/credit_default/fred.py`).

### Target definition
- **`default_flag = 1`** if the loan status is *Charged Off*.
- **`default_flag = 0`** if the loan status is *Fully Paid*.
- This is a **lifetime default** target over the 36-month term, not a 12-month PD.

### Sample filters and why
| Filter | Reason |
|---|---|
| Only *Fully Paid* or *Charged Off* loans | Every loan has a known final outcome. Loans still being repaid have no outcome yet. |
| Only 36-month loans | Keeps the default window consistent across loans. |
| Only loans issued **2012–2015** | See the censoring issue below. Earlier years have few loans and reflect a different stage of Lending Club's business. |

### The censoring trap
Default rates by issue year, among completed 36-month loans:

| Year | Loans | Default rate |
|---|---|---|
| 2012 | 43,470 | 13.6% |
| 2013 | 100,422 | 12.3% |
| 2014 | 162,570 | 13.7% |
| 2015 | 283,026 | 14.9% |
| 2016 | 232,353 | 19.8% |
| 2017 | 128,538 | 20.0% |
| 2018 | 41,268 | 13.2% |

The jump in 2016–2017 is largely an artifact. The data ends in late 2018, so 36-month loans issued in 2016 or later had **not finished** by then. Among those vintages, the loans with a final outcome are disproportionately **early defaults**, which inflates the default rate. To avoid this censoring bias, the sample is restricted to vintages whose 36-month loans had **fully matured** by the end of 2018, meaning loans issued by December 2015.

### Out-of-time split
| Set | Vintages | Loans | Default rate |
|---|---|---|---|
| **Train** | 2012–2014 | 306,462 | 13.2% |
| **Test (out-of-time)** | 2015 | 283,026 | 14.9% |

---

## 3. Approach

### Pipeline
```
Kaggle API → DuckDB (SQL) → features → models → evaluation → recalibration
FRED API  → DuckDB (SQL join) → macro comparison
                                → explainability
```

1. **Download** (`download.py`): pulls only the accepted-loans file through the Kaggle API.
2. **Load and filter with SQL** (`sql/01_build_loan_table.sql`, `data.py`): DuckDB reads the compressed CSV, builds the target, applies the sample filters, and selects origination-time features.
3. **Feature engineering** (`features.py`): cleaning, engineered features, imputation, and the out-of-time split.
4. **Training** (`train.py`): logistic regression benchmark and XGBoost challenger.
5. **Evaluation** (`evaluate.py`): AUC, Gini, KS, calibration, ROC, and PSI.
6. **Recalibration** (`recalibrate.py`): intercept adjustment using the most recent fully observed vintage.
7. **Macro data** (`fred.py`, `sql/02_add_macro.sql`): pulls FRED series and joins them to loans in SQL.
8. **Macro comparison** (`compare_macro.py`): base vs. macro-enhanced logistic regression.
9. **Explainability** (`explain.py`): odds ratios, SHAP values, partial dependence, and monotonic constraints.

### Features
**Used (all known at origination):**
- Loan amount, debt-to-income (capped at the 99th percentile), FICO score (lower bound of range), revolving utilization, delinquencies in the past 2 years, inquiries in the past 6 months, open accounts, public records
- **Engineered:** log annual income, employment length in years, loan-to-income ratio
- **Missing indicators** for employment length, DTI, and revolving utilization, since missingness can itself carry risk information
- **Categorical:** home ownership and loan purpose, one-hot encoded with a dropped reference category (**mortgage holders** and **car loans**). Rare categories (under 1%) are grouped, and categories unseen in training are mapped to that infrequent group.
- **Macro (comparison model only):** national unemployment, federal funds rate, state unemployment, and 12-month change in state unemployment

**Deliberately excluded:**
- **Post-origination fields** such as total payments, recoveries, and last payment date. These are only known after the loan plays out and would cause **data leakage**.
- **Lending Club's grade and interest rate.** These are outputs of Lending Club's own risk model. Including them would raise AUC but make this model largely replicate Lending Club's pricing. Excluding them keeps the model **independent**.

### Leakage controls
- Only origination-time features, enforced by automated tests.
- Imputation medians are computed on the **training data only** and then applied to the test set.
- Preprocessing (scaling, encoding) is wrapped in a scikit-learn `Pipeline` / `ColumnTransformer`, so the same fitted transformations are applied consistently.
- Recalibration uses only data that would be available at prediction time (the 2014 vintage), never the 2015 test outcomes.
- Macro values use the **month before origination**, since monthly economic data is published with a lag.

### Models
**Logistic regression (benchmark)**
- Standardized numeric features and one-hot encoded categories.
- **No class weights.** With a 13% default rate, the imbalance is moderate, and class weighting would distort the predicted probabilities. Because this is a PD model, calibrated probabilities take priority.

**XGBoost (challenger)**
- Shallow trees (`max_depth=4`), learning rate 0.05, row and column subsampling of 0.8.
- **Time-aware early stopping**: trained on 2012–2013 and validated on 2014, stopping after 50 rounds without AUC improvement. Training stopped at **207 trees**.

**Monotonic XGBoost (constrained challenger)**
- The same configuration, with monotonic constraints so key risk drivers can only move predicted risk in the economically sensible direction.

---

## 4. Results

All results are on the **2015 out-of-time test set** (283,026 loans).

### Discrimination
| Model | AUC | Gini | KS | Mean predicted PD | Actual default rate |
|---|---|---|---|---|---|
| Logistic regression | 0.655 | 0.310 | 0.223 | 13.2% | 14.9% |
| XGBoost | 0.655 | 0.309 | 0.224 | 12.7% | 14.9% |
| Monotonic XGBoost | 0.654 | — | — | — | 14.9% |

![ROC curve](reports/figures/roc.png)

**Interpretation:**
- All models separate risk meaningfully, though modestly, which is expected for a model built only on application data without Lending Club's grade.
- **XGBoost adds essentially nothing over logistic regression.** The relationships in these features appear largely monotonic and close to linear in log-odds, so the added complexity of boosting does not pay off. Given equal performance, the **logistic regression is preferred** for its interpretability, stability, and ease of validation, which are the qualities banks and regulators value in PD models.

### Calibration (logistic regression, by risk decile)
| Decile | Loans | Predicted | Actual |
|---|---|---|---|
| 0 (lowest risk) | 28,303 | 4.17% | 4.28% |
| 1 | 28,303 | 6.81% | 7.60% |
| 2 | 28,302 | 8.54% | 9.76% |
| 3 | 28,303 | 10.04% | 11.36% |
| 4 | 28,302 | 11.50% | 13.55% |
| 5 | 28,303 | 13.05% | 14.85% |
| 6 | 28,302 | 14.77% | 17.51% |
| 7 | 28,303 | 16.86% | 19.23% |
| 8 | 28,302 | 19.79% | 22.59% |
| 9 (highest risk) | 28,303 | 26.55% | 28.13% |

![Calibration](reports/figures/calibration.png)

**Interpretation:**
- **Ranking holds up:** actual default rates rise monotonically across deciles, from 4.3% to 28.1%.
- **Levels are too low:** the model underpredicts default in nearly every decile, by roughly 1 to 3 percentage points. Overall, it predicts 13.2% against an actual 14.9%. Its predictions are anchored to the training period's lower default rate.

### Population stability
| Comparison | Score PSI |
|---|---|
| Train (2012–2014) vs. test (2015) | **0.001** |

A PSI below 0.10 is conventionally considered stable. At 0.001, the score distribution is essentially unchanged.

---

## 5. Key Finding: Concept Drift, Not Data Drift

The evaluation produces two results that look contradictory at first:

1. The **population did not change**. The score PSI is 0.001, so 2015 borrowers look almost identical to 2012–2014 borrowers on the features the model uses.
2. **Outcomes did change**. The default rate rose from 13.2% to 14.9%, and the model underpredicted risk across the board.

Together, these point to **concept drift**: the *relationship* between borrower characteristics and default shifted, rather than the *mix* of borrowers. Borrowers with the same profile were simply more likely to default in the 2015 vintage, plausibly reflecting vintage-level effects such as underwriting changes, competitive conditions in the lending market, or the credit cycle. The macro analysis in Section 7 supports the credit-cycle explanation.

**Why this matters for model monitoring:** a monitoring program that relied only on PSI or other input-stability checks would have reported the model as stable. The deterioration is only visible through **calibration and performance monitoring** against realized outcomes. Effective credit model monitoring needs both:

| Monitoring type | What it detects | What it found here |
|---|---|---|
| Population stability (PSI) | Data drift: changes in inputs | Nothing (PSI = 0.001) |
| Calibration and performance | Concept drift: changes in relationships | Systematic underprediction |

---

## 6. Recalibration

When scoring 2015 loans, the most recent fully observed vintage would be **2014**. The model's baseline was recalibrated using only that vintage, through **calibration-in-the-large**: a single intercept shift in log-odds, which preserves the ranking and AUC while adjusting the probability levels.

On the 2014 vintage, the model predicted **13.50%** against an actual **13.73%**, giving an intercept shift of **+0.020** in log-odds.

| | Mean PD |
|---|---|
| Actual 2015 default rate | 14.89% |
| Before recalibration | 13.21% |
| After recalibration (2014 vintage) | 13.43% |

![Calibration before and after](reports/figures/calibration_recalibrated.png)

**Interpretation:** recalibrating on the latest available data narrowed the gap only slightly. The 2014 vintage was already part of the training data, and its default rate was close to the model's predictions, so there was little to correct. The remaining gap of roughly 1.5 percentage points reflects concept drift that was **not visible in any default outcomes available before 2015**. Recalibration on recent data helps, but it cannot anticipate a shift that has not yet appeared in realized defaults.

---

## 7. Macroeconomic Features (FRED API)

To test whether economic conditions explain the 2015 underprediction, macro data was pulled from the **FRED API** and joined to each loan in SQL (`fred.py`, `sql/02_add_macro.sql`):

- **National unemployment rate** and **federal funds rate**
- **State unemployment rate** for the borrower's state (all 50 states plus DC)
- **12-month change in state unemployment**, computed with a SQL window function (`LAG ... OVER (PARTITION BY state ORDER BY date)`)

All macro values use the **month before origination**, since monthly data is published with a lag. Only 0.70% of loans lacked a state match; these were imputed with training medians.

### Results (2015 out-of-time, logistic regression)

| Model | AUC | Mean predicted PD | Actual default rate |
|---|---|---|---|
| Base | 0.655 | 13.21% | 14.89% |
| Base + recalibration | 0.655 | 13.43% | 14.89% |
| **Base + macro features** | **0.655** | **14.09%** | **14.89%** |

Macro features **did not improve ranking**, but they **cut the calibration gap roughly in half**, from 1.68 to 0.80 percentage points, more than recalibration achieved.

### Calibration by decile

| Decile | Base predicted | Macro predicted | Actual |
|---|---|---|---|
| 0 (lowest risk) | 4.17% | 4.52% | 4.28% |
| 1 | 6.81% | 7.35% | 7.60% |
| 2 | 8.54% | 9.19% | 9.76% |
| 3 | 10.04% | 10.78% | 11.36% |
| 4 | 11.50% | 12.32% | 13.55% |
| 5 | 13.05% | 13.95% | 14.85% |
| 6 | 14.77% | 15.77% | 17.51% |
| 7 | 16.86% | 17.96% | 19.23% |
| 8 | 19.79% | 21.02% | 22.59% |
| 9 (highest risk) | 26.55% | 28.00% | 28.13% |

### Macro odds ratios

| Macro feature | Odds ratio (per 1 SD) |
|---|---|
| State unemployment | 1.034 |
| 12-month change in state unemployment | 1.007 |
| Fed funds rate | 1.023 |
| National unemployment | **0.951** |

### A counterintuitive sign, and why it matters

National unemployment has a **negative** relationship with default: loans originated when unemployment was higher defaulted *less*. This is consistent with a credit-cycle effect. Lenders tighten underwriting when unemployment is high and loosen it late in a recovery, so the variable is likely acting as a **proxy for underwriting standards and the credit cycle**, not as a direct driver of default.

That interpretation also explains the improved calibration, and it supports the concept drift finding: the 2015 shift appears tied to the credit cycle rather than to changes in borrower characteristics.

**However, this feature should not be used in production as is.** National unemployment is nearly constant within each month, so its coefficient was estimated from roughly 36 monthly observations covering a single phase of the economic cycle. In a recession, the model would predict *lower* risk as unemployment rose, which is exactly the wrong response at the worst time. A model validator would challenge this relationship.

**Conclusion:** state unemployment is a sensible cross-sectional feature worth keeping. The national result is best treated as **evidence of a cycle-driven shift**, pointing toward explicit vintage or underwriting-cycle modeling, or training on data that spans a full credit cycle, rather than as a production feature.

---

## 8. Explainability

Explainability follows a **Show, Plot, Constrain, Compare** approach.

### Compare: logistic regression odds ratios
Numeric features are standardized, so their odds ratios are **per one standard deviation**. Categorical odds ratios are relative to the reference groups: **car loans** for purpose and **mortgage holders** for home ownership.

| Feature | Odds ratio | Interpretation |
|---|---|---|
| Purpose: small business | 2.40 | About 2.4x the odds of default vs. car loans |
| Purpose: medical | 1.48 | Higher risk than car loans |
| Purpose: rare categories (grouped) | 1.41 | Higher risk than car loans |
| Home ownership: rent | 1.30 | About 30% higher odds than mortgage holders |
| Inquiries in the past 6 months | 1.18 | More recent credit-seeking, higher risk |
| Loan-to-income | 1.16 | Larger loans relative to income, higher risk |
| Log annual income | 0.85 | Higher income, lower risk |
| Purpose: credit card | 0.84 | Slightly lower risk than car loans |
| FICO score | 0.70 | Each 1 SD higher FICO cuts the odds of default by about 30% |

All signs match economic intuition.

### Show: SHAP values (XGBoost)
SHAP values were computed natively with XGBoost (`pred_contribs`) on a 5,000-loan sample of the 2015 test set. Values are in log-odds units.

![SHAP summary](reports/figures/shap_summary.png)
![SHAP importance](reports/figures/shap_importance.png)

**Reason codes for the highest-risk loan in the sample**, the kind of output that supports adverse action notices:
1. High loan-to-income ratio
2. Low FICO score
3. High debt-to-income
4. A rare loan purpose
5. High revolving utilization

### Plot: partial dependence
Partial dependence shows how average predicted risk changes as debt-to-income, FICO, revolving utilization, and loan-to-income change.

![Partial dependence](reports/figures/partial_dependence.png)

### Constrain: monotonic constraints
XGBoost was retrained with monotonic constraints so that higher DTI, revolving utilization, inquiries, delinquencies, public records, and loan-to-income can only **increase** predicted risk, and higher FICO and income can only **decrease** it.

| Model | Out-of-time AUC |
|---|---|
| Unconstrained XGBoost | 0.655 |
| Monotonic XGBoost | 0.654 |

The constraints cost essentially **nothing** in accuracy while guaranteeing intuitive, defensible behavior, so the constrained version is preferred whenever a tree-based model is used.

---

## 9. Engineering and Reproducibility

### Automated tests
**21 tests** (`tests/`) run locally with `make test` and automatically on every push through **GitHub Actions**:

| Test file | What it checks |
|---|---|
| `test_leakage.py` | No post-origination fields in features or SQL, Lending Club model outputs excluded, target not used as a feature, and a strict time split with no overlap |
| `test_metrics.py` | PSI is zero for identical distributions, flags large shifts, and is never negative; KS and AUC/Gini behave correctly; calibration tables account for every loan |
| `test_recalibration.py` | The intercept shift hits its target, preserves ranking, does nothing when already calibrated, and keeps PDs between 0 and 1 |
| `test_features.py` | Employment length parsing, missing indicators, safe handling of zero income, and no modification of input data |

Tests use small synthetic data, so they run in seconds without downloading the dataset.

### Run archives
Every full pipeline run (`make all`) is saved to its own timestamped folder:

```
reports/
├── figures/                  # latest results (linked from this README)
└── runs/
    └── YYYYMMDD_HHMMSS/
        ├── run_info.txt      # run ID, Git commit, and the full config used
        ├── run_log.txt       # all printed output from every step
        └── ...               # every figure and table from that run
```

This makes every result traceable to the exact code and settings that produced it.

### Other practices
- **Pinned dependencies** in `requirements.txt` for exact reproducibility.
- **One-command pipeline** with `make all`, plus individual steps (`make train`, `make explain`, and so on).
- **A model card** (`docs/model_card.md`) summarizing intended use, performance, limitations, and a monitoring plan with review triggers.
- **Credentials kept out of the code**: the Kaggle token lives in `~/.kaggle/`, and the FRED key in an untracked `.env` file.

---

## 10. Limitations

- **Modest discrimination.** An AUC of 0.655 reflects an independent model using application data only. Lending Club's grade and interest rate, which were deliberately excluded, carry substantial predictive information. Macro features did not improve ranking.
- **Residual miscalibration.** The base model underpredicts 2015 defaults by about 1.7 percentage points. Recalibration reduced this only slightly, and the macro-enhanced model still underpredicts by about 0.8 points.
- **Untrustworthy national macro relationship.** National unemployment's coefficient reflects a single phase of the credit cycle, estimated from roughly 36 monthly observations, and has a counterintuitive sign that would mislead the model in a downturn.
- **Recalibration design.** The 2014 vintage used for recalibration was also part of the training data. A stricter design would train on 2012–2013 only and recalibrate on 2014 as a true holdout.
- **Single out-of-time vintage.** Results are based on one test year. Performance across multiple future vintages and economic conditions has not been assessed.
- **Lifetime target, not a 12-month PD.** The model predicts default over the 36-month term, which differs from the 12-month PDs common in some regulatory and accounting contexts. It is closer in spirit to a lifetime loss view, but it is not a full CECL model.
- **Approved loans only.** The data includes only loans Lending Club approved, so the model has not seen rejected applicants. Using it for underwriting decisions would require addressing this selection bias (reject inference).
- **Limited feature set and tuning.** Features are a compact set of origination variables, and XGBoost hyperparameters were set sensibly rather than tuned through a full search.
- **Platform-specific data.** Lending Club's borrower base, underwriting, and history may not generalize to other lenders or loan products.

---

## 11. Further Development

**Credit cycle and macro modeling**
- Model the credit cycle explicitly, for example with **vintage effects** or underwriting-cycle indicators, rather than relying on national unemployment as a proxy.
- Train on data that spans a **full credit cycle**, such as the Freddie Mac Single-Family Loan-Level Dataset covering 2008 and 2020, so macro relationships are estimated across both downturns and recoveries.
- Test macro conditions **over the life of the loan** in a stress-testing or scenario framework, rather than at origination only.

**Calibration and monitoring**
- A **stricter recalibration design**: train on 2012–2013 and recalibrate on a 2014 holdout.
- **Feature-level PSI** to monitor each input, not just the overall score.
- **Rolling out-of-time evaluation** across multiple vintages to track performance stability over time.

**Modeling**
- A **Weight of Evidence (WoE) scorecard** version of the logistic regression, the classic approach in retail credit.
- A **survival or discrete-time hazard model**, which handles loans that have not yet matured directly, rather than excluding later vintages.
- **Hyperparameter tuning** with time-aware cross-validation.
- A **full expected loss** estimate (PD × LGD × EAD) using recovery data, as a step toward a CECL-style framework.

**Engineering**
- Share a single preprocessing definition between `train.py` and `compare_macro.py`.
- A **Streamlit dashboard** for exploring predictions, calibration, and monitoring results.

---

## 12. Project Structure

```
credit_default_model_project/
├── .github/workflows/
│   └── tests.yml              # runs the test suite on every push
├── data/                      # not tracked in Git
│   ├── raw/                   # downloaded Lending Club data
│   └── processed/             # parquet tables and test predictions
├── docs/
│   └── model_card.md          # intended use, performance, limitations, monitoring plan
├── models/                    # saved models (not tracked in Git)
├── reports/
│   ├── figures/               # latest figures and tables
│   └── runs/                  # timestamped archives of each full run
├── sql/
│   ├── 01_build_loan_table.sql
│   └── 02_add_macro.sql       # joins FRED macro data to loans (CTE + window function)
├── src/credit_default/
│   ├── config.py              # paths, target definition, split dates, seed
│   ├── download.py            # Kaggle API download
│   ├── data.py                # SQL load into DuckDB
│   ├── features.py            # cleaning, features, out-of-time split
│   ├── train.py               # logistic regression and XGBoost
│   ├── evaluate.py            # AUC, Gini, KS, calibration, ROC, PSI
│   ├── recalibrate.py         # intercept recalibration on the 2014 vintage
│   ├── fred.py                # FRED API pull and SQL macro join
│   ├── compare_macro.py       # base vs. macro-enhanced model comparison
│   └── explain.py             # odds ratios, SHAP, partial dependence, monotonic constraints
├── tests/
│   ├── test_features.py
│   ├── test_leakage.py
│   ├── test_metrics.py
│   └── test_recalibration.py
├── tools/
│   └── export_codebase.py     # prints the project tree and saves code snapshots
├── Makefile                   # one-command pipeline and run archiving
├── pytest.ini
├── requirements.txt           # pinned dependencies
├── setup_env.py               # creates the virtual environment
└── README.md
```

---

## 13. How to Run

**1. Set up the environment**
```bash
python3 setup_env.py
source venv/bin/activate
```

**2. Add credentials**
- **Kaggle:** create a Kaggle API token and store it in `~/.kaggle/` with permissions set to `600`.
- **FRED:** request a free API key from FRED and save it in a `.env` file in the project root as `FRED_API_KEY=your_key`. The `.env` file is excluded from Git.

**3. Download the data (once)**
```bash
make download
```

**4. Run the full pipeline**
```bash
make all
```
This builds the data, trains and evaluates the models, recalibrates, adds macro features, runs explainability, and archives everything to `reports/runs/<timestamp>/`.

**5. Run the tests**
```bash
make test
```

Individual steps can also be run on their own: `make data`, `make features`, `make train`, `make evaluate`, `make recalibrate`, `make macro`, and `make explain`.

---

## 14. Tech Stack

Python · SQL (DuckDB) · pandas · NumPy · SciPy · scikit-learn · XGBoost · SHAP · Matplotlib · pytest · GitHub Actions · Make · Kaggle API · FRED API · Parquet
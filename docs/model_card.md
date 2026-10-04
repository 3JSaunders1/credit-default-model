# Model Card: Credit Default Prediction Model

## Intended use
Research and portfolio demonstration of probability of default (PD) modeling and
validation practices. Not intended for real credit decisions.

## Model
- **Primary:** logistic regression
- **Challengers:** XGBoost, monotonic XGBoost, and a WoE scorecard
- **Target:** lifetime charge-off on 36-month Lending Club loans
- **Categorical reference groups:** mortgage holders (home ownership) and car loans (purpose)

## Data
- Lending Club accepted loans issued 2012–2015, fully matured by the end of 2018
- **Train:** 2012–2014 (306,462 loans)
- **Out-of-time test:** 2015 (283,026 loans)
- Optional macro features from the FRED API (national and state unemployment, federal funds rate)

## Performance (2015 out-of-time)
- AUC 0.655 (logistic regression) and 0.660 (XGBoost, trained on the same data); Gini 0.31–0.32, KS 0.22–0.23
- Mean predicted PD 13.2% vs. actual 14.9%
  - 13.4% after recalibration on the 2014 vintage (13.3% under a strict design with 2014 held out)
  - 14.1% with macro features
- Score PSI 0.001
- Monotonic XGBoost AUC 0.659 vs. 0.660 unconstrained
- WoE scorecard AUC 0.651 using 8 of 16 features
- Time-aware tuning: 21 XGBoost configurations within 0.003 AUC; settings are not the bottleneck

## Known limitations
- Modest discrimination; Lending Club grade and interest rate deliberately excluded
- Underpredicts the 2015 vintage due to concept drift
- National unemployment shows a cycle-driven sign and is unsafe for production
- Approved loans only (no reject inference)
- Single out-of-time vintage

## Monitoring plan
- **Population stability:** score and feature-level PSI (watch above 0.10, investigate above 0.25)
- **Performance:** AUC, Gini, and KS by vintage
- **Calibration:** predicted vs. actual default rates by risk decile
- **Review triggers:** PSI above 0.25, an AUC drop of more than 0.03, or a calibration gap above 1 percentage 
point

## Explainability
Logistic odds ratios, SHAP reason codes, partial dependence plots, and monotonic constraints.

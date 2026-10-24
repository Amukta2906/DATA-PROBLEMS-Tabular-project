# Customer Churn Prediction: Synthetic vs. Real Telecom Data

**DATA 4380 Group Project, Fall 2026**
Amukta Chaganty · Uyen Nguyen · Prakriti Kattel

## Overview
Customer churn is when a customer leaves a service. We built machine learning pipelines to
predict which telecom customers will churn, so a company could target retention offers
before customers leave.

We first built a full pipeline on the **IBM Telco** dataset, which is synthetic sample data
for a fictional company. Every improvement stalled near ROC-AUC 0.84, so we repeated the
work on **Cell2Cell**, real customer data from a US wireless carrier, to see how real-world
data changes the problem.

## Datasets
| | IBM Telco | Cell2Cell |
|---|---|---|
| Source | IBM sample data (fictional company) | Real US wireless carrier (Teradata Center for CRM, Duke University) |
| Customers | 7,043 | 51,047 (training file) |
| Columns | 21 | 58 |
| Churn rate | 26.5% | 28.8% |
| Missing data | 11 blank TotalCharges values | Real blanks (under 2%) plus hidden "unknown" codes (25 to 57% in some columns) |

## Approach
**Shared pipeline (both datasets)**
- Stratified 80/20 train/test split before any exploration; the test set is scored only once, at the end
- All preprocessing (median imputation, StandardScaler, one-hot encoding) inside scikit-learn
  pipelines, so every cross-validation fold is fit on its own training rows and nothing leaks
- 5-fold stratified cross-validation for every modeling decision
- Models: Dummy, Logistic Regression, Decision Tree, Random Forest, XGBoost
- Class imbalance: class weights, SMOTE (inside training folds only) and decision-threshold tuning
- Feature selection: ANOVA F-test with SelectKBest inside the pipeline (plus Chi-square on Telco)
- Hyperparameter tuning: RandomizedSearchCV
- Metrics: recall, precision, F1, ROC-AUC and PR-AUC (accuracy is misleading with imbalanced
  classes); top-decile lift on Cell2Cell

**Telco-specific**
- TotalCharges stored as text (11 blank cells): converted to numbers, filled with the median
- Merged "No internet service" / "No phone service" into "No" (38 encoded columns)

**Cell2Cell-specific**
- Hidden missing values (AgeHH1 = 0, IncomeGroup = 0, HandsetPrice = "Unknown") treated as
  missing; median fill plus a 0/1 "was missing" flag
- ServiceArea (747 codes) reduced to 57 markets; CreditRating converted to an ordinal 1 to 7
- 13 engineered features: phone age, contract-end window, dropped calls and care calls per
  minute, revenue per minute, usage-drop flags
- Compared imputing vs. dropping rows with missing values

## Results (test set, scored once)
| Dataset | Final model | Threshold | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| IBM Telco | Logistic Regression, class weights, top 15 ANOVA features | 0.541 | 0.832 | 0.620 | 0.74 | 0.52 | 0.61 |
| Cell2Cell | Tuned XGBoost, class weights, all features + 13 engineered | 0.421 | 0.684 | 0.464 | 0.80 | 0.37 | 0.50 |

Cell2Cell top-decile lift: **1.94** (the 10% of customers ranked riskiest churn at 55%, vs. 29% overall).

## Key findings
- **Accuracy is misleading:** a model that always predicts "stayed" is 71 to 74% accurate and catches no churners.
- **The cutoff matters most:** class weights and a tuned threshold raised recall from about 5% to 80%
  on Cell2Cell. They change who gets flagged, not how well customers are ranked.
- **Tuning, SMOTE and feature selection barely moved ROC-AUC** on either dataset.
- **Real data is harder:** the same pipeline scores ROC-AUC 0.84 on synthetic Telco but 0.68 on real
  Cell2Cell. The limit is the information in the data, not the model.
- **Top churn signals:** Telco: contract type, tenure, internet service and payment method.
  Cell2Cell: falling minutes used, months in service, revenue per minute and phone age.

## Repository structure
├── README.md
├── requirements.txt
├── telco_churn.ipynb # IBM Telco: cleaning, EDA, baselines, tuning, final test
└── cell2cell_churn.ipynb # Cell2Cell: cleaning, EDA, feature engineering, models, final test


## How to run
1. Install the dependencies (or use Google Colab, which already has most of them):

pip install -r requirements.txt

2. Download the data from Kaggle and put the CSV in the same folder as the notebook:
   - IBM Telco: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
     (file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`)
   - Cell2Cell: https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom
     (file: `cell2celltrain.csv`; the holdout file has no churn labels, so we do not use it)
3. Run all cells from top to bottom. Telco takes about 5 minutes; Cell2Cell takes about
   20 to 30 minutes (Random Forest and XGBoost tuning on 41k rows).

All random seeds are fixed (`random_state=42`), so results are reproducible.

## Environment
Python 3.10+, pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imbalanced-learn.
Developed and run in Google Colab.

## References
- Omar, A. S., Ogada, K., and Tole, K. (2026). A hybrid feature selection framework for
  dimensionality-optimized customer churn prediction. *Scientific Reports*, 16, 25758.
  https://doi.org/10.1038/s41598-026-52160-z
- IBM Telco Customer Churn dataset (Kaggle).
- Cell2Cell churn dataset, Teradata Center for CRM at Duke University (Kaggle).

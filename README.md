# Customer Churn Prediction: Synthetic vs. Real Telecom Data

DATA 4380 Group Project, Fall 2026 · Uyen Nguyen, Amukta Chaganty, Prakriti Kattel

## Overview
We predict which telecom customers will churn. We first built a full pipeline on the
IBM Telco dataset (synthetic), then repeated it on Cell2Cell, real data from a US
wireless carrier (51,047 customers), to see how real-world mess changes the results.

## Approach
- Stratified 80/20 train/test split; the test set is scored only once at the end
- All preprocessing (imputation, scaling, one-hot encoding) inside scikit-learn pipelines
- 5-fold stratified cross-validation for every modeling decision
- Models: Dummy, Logistic Regression, Decision Tree, Random Forest, XGBoost
- Class weights, SMOTE comparison, threshold tuning, ANOVA feature selection
- Cell2Cell: hidden "unknown" codes treated as missing, 13 engineered features
- Metrics: recall, F1, ROC-AUC, PR-AUC, top-decile lift (accuracy is misleading here)

## Results (test set)
| Dataset | Final model | ROC-AUC | Recall | F1 |
|---|---|---|---|---|
| IBM Telco | [fill in] | [ ] | [ ] | [ ] |
| Cell2Cell | XGBoost + class weights + engineered features | 0.684 | 0.80 | 0.50 |

## Repository structure
- `Telco_Churn.ipynb`: IBM Telco pipeline
- `Cell2Cell_Churn.ipynb`: Cell2Cell pipeline
- `requirements.txt`: Python dependencies

## How to run
1. Install dependencies: `pip install -r requirements.txt` (or run in Google Colab)
2. Download the data from Kaggle and place the CSV next to the notebook:
   - IBM Telco: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
     (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
   - Cell2Cell: https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom
     (`cell2celltrain.csv`, the training file only)
3. Run all cells from top to bottom. The Cell2Cell notebook takes about 20–30 minutes.

## References
Omar, A. S., Ogada, K., and Tole, K. (2026). A hybrid feature selection framework for
dimensionality-optimized customer churn prediction. *Scientific Reports*, 16, 25758.

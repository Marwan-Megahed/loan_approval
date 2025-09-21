# loan_approval
Loan default prediction with preprocessing, SMOTE, EDA, and multiple classifiers.
# Loan Data Analysis and Prediction Notebook

## Overview
This Jupyter notebook (`loan_data_v3.ipynb`) performs exploratory data analysis (EDA), data preprocessing, and machine learning modeling on a loan dataset to predict whether a loan is fully paid or not. The dataset includes features related to borrowers' credit profiles, loan purposes, and financial metrics. The goal is to build and evaluate classification models to identify loans at risk of not being fully paid.

Key tasks include:
- Data loading and cleaning.
- Visualizations for insights (e.g., distributions, correlations, approval ratios).
- Handling class imbalance using SMOTE.
- Training and evaluating multiple classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost).
- Hyperparameter tuning and model performance metrics (accuracy, precision, recall, F1-score, ROC curves).

## Dataset
- **File**: `loan_data.csv` (not included in the notebook; assumed to be in the same directory).
- **Source**: Not specified (likely from LendingClub or a similar platform).
- **Features** (14 columns):
  - `credit.policy`: 1 if the customer meets the credit underwriting criteria; 0 otherwise.
  - `purpose`: Categorical (e.g., credit_card, debt_consolidation).
  - `int.rate`: Interest rate of the loan (as a proportion).
  - `installment`: Monthly installment amount.
  - `log.annual.inc`: Log of self-reported annual income.
  - `dti`: Debt-to-income ratio.
  - `fico`: FICO credit score.
  - `days.with.cr.line`: Days with a credit line.
  - `revol.bal`: Revolving balance.
  - `revol.util`: Revolving line utilization rate.
  - `inq.last.6mths`: Number of inquiries in the last 6 months.
  - `delinq.2yrs`: Number of delinquencies in the past 2 years.
  - `pub.rec`: Number of derogatory public records.
  - **Target**: `not.fully.paid`: 1 if not fully paid; 0 otherwise.

- **Size**: 9578 entries.
- **Imbalance**: The target class is imbalanced (more fully paid loans).

**Data Quality**: No missing values or duplicates in the dataset.

## Requirements
- **Python Version**: 3.12.3 (or compatible).
- **Libraries**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly.express
  - imblearn (for SMOTE)
  - scikit-learn (for models, preprocessing, metrics, GridSearchCV)
  - xgboost (for XGBoost classifier)
  - warnings (optional, for suppressing warnings)

### Installation
Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn plotly imbalanced-learn scikit-learn xgboost

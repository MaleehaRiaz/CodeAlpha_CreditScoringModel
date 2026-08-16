# Credit Scoring Model Project

# What Exactly Does This Project Do?
This project predicts whether a person is likely to default on a loan (miss payments seriously) within the next two years, based on their financial history. Banks and lenders use this kind of model to decide whether someone is a good credit risk.

# Dataset
Give Me Some Credit is a real dataset from a 2011 Kaggle competition, containing 150,000 anonymized borrower records with features like income, debt ratio, credit utilization, and past late payments.

# Project Workflow
1. Data Cleaning: Fixed an invalid age value (age = 0), corrected placeholder glitch values (96/98) in the late-payment columns, and filled in missing income and dependents values
2. Feature Engineering: Combined the three late-payment columns into a single TotalTimesLate feature, which correlated more strongly with default than any single original feature. Also created HasBeenLate, TotalCreditLines, and IncomePerDependent
3. Modeling: Trained and compared two models:
   i- Logistic Regression (baseline)
   ii- Random Forest
4. Evaluation: Used Precision, Recall, F1-Score, and ROC-AUC (not just accuracy, since the target is heavily imbalanced; only 6.7% of people in the dataset actually defaulted)
5. Extra Work: Added 5-fold cross-validation, hyperparameter tuning (GridSearchCV), an XGBoost comparison model, SHAP-based interpretability, and decision threshold tuning.

Results

Logistic Regression
Random Forest

5-fold cross-validation confirmed that the Random Forest result is stable: mean ROC-AUC = 0.861 (± 0.003)
For reference, the actual winning submission in the original 2011 Kaggle Competition on this dataset scored an ROC-AUC of about 0.869, so this model's performance is competitive with the best-known public result.

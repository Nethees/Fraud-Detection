# Fraud Detection Project

End-to-end fraud detection pipeline built on the Kaggle "Credit Card Fraud
Detection" dataset. Trains and compares Logistic Regression, Random Forest,
and XGBoost on a highly imbalanced classification problem, evaluated on
PR-AUC rather than accuracy since fraud is a rare event.

## Setup

1. `pip install -r requirements.txt`
2. Download `creditcard.csv` from Kaggle (mlg-ulb/creditcardfraud) and place
   it in this folder.
3. `python main.py`

## Structure

- `config.py` — paths, constants, model hyperparameters
- `data_loader.py` — loading and exploratory analysis
- `preprocessing.py` — scaling and train/test split
- `models.py` — model definitions and training
- `evaluate.py` — metrics, confusion matrix, ROC/PR curves
- `feature_importance.py` — feature importance plots
- `main.py` — orchestrates the pipeline
- `outputs/` — saved plots and the model comparison CSV land here

## Notes

Class imbalance is handled via `class_weight="balanced"` for Logistic
Regression and Random Forest, and `scale_pos_weight` for XGBoost, rather
than oversampling with SMOTE, since synthetic points on PCA-transformed
features can be unrealistic.

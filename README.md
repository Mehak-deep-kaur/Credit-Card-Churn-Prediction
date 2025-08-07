# Credit Card Churn Prediction Capstone Project

## Project Overview
This project predicts credit card customer churn using machine learning techniques. We built a Random Forest classifier with clean, processed data, and performed feature engineering and scaling. The pipeline simulates production deployment by integrating with a MySQL database (Aurora MySQL simulation).

## Business Problem
Churn leads to loss of revenue for banks. Early prediction helps in retaining valuable customers via targeted actions by analyzing customer behavior and demographics.

## Dataset
- File: `data/raw/exl_credit_card_churn_data.csv`  
- Features include customer demographics, card usage, and churn label (0 = Non-churn, 1 = Churn).

## Key Features
- Data cleaning & preprocessing (missing values, inconsistencies, outliers)
- One-hot encoding for categorical variables
- Feature engineering: BalanceSalaryRatio, TenurePerProduct, IsHighValueCustomer
- MinMax Scaling of numerical features
- Random Forest classification with hyperparameter tuning



## How To Run
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
   3. Place the raw dataset at `data/raw/exl_credit_card_churn_data.csv`
4. Run the full pipeline:
## Outputs
- Cleaned and processed dataset (`churn_cleaned.csv`)
- Trained Random Forest model (`random_forest.pkl`)
- Scaler model (`scaler_model.pkl`)
- Model evaluation reports and prediction outputs printed during runtime
- # Notes
- Categorical features are one-hot encoded as per project requirements.  
- AWS Aurora MySQL setup is simulated locally in this repo and can be adapted for real AWS deployment.


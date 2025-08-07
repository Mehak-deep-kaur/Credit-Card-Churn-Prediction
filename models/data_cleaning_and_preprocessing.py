# scripts/data_cleaning_and_preprocessing.py
import pandas as pd
import numpy as np

def understand_data(df):
    print("\n--- Initial Data Understanding ---")
    print("DataFrame Info:")
    df.info()
    
    print("\nNull Value Count:")
    print(df.isnull().sum())
    
    print("\nUnique Values in Selected Columns:")
    for col in ['Gender', 'HasCrCard', 'IsActiveMember', 'Churn']:
        print(f"'{col}': {df[col].unique()}")

def clean_data(df):
    # --- Data Cleaning ---
    
    # Clean & Impute 'Gender' column
    df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize().replace('Nan', pd.NA)
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

    # Clean & Impute 'Age' column
    df['Age'] = df['Age'].apply(lambda x: np.nan if pd.isna(x) or x < 0 or x > 100 else x)
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Impute numeric columns
    for col in ['Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Handle 'HasCrCard' and 'IsActiveMember' columns
    df['HasCrCard'] = df['HasCrCard'].replace({'Yes': 1, 'No': 0, '1': 1, '0': 0})
    df['IsActiveMember'] = df['IsActiveMember'].replace({'Yes': 1, 'No': 0, '1': 1, '0': 0})
    df['HasCrCard'] = pd.to_numeric(df['HasCrCard'], errors='coerce').fillna(0)
    df['IsActiveMember'] = pd.to_numeric(df['IsActiveMember'], errors='coerce').fillna(0)

    # Clean 'Churn' column and filter data
    df['Churn'] = df['Churn'].astype(str).str.strip().str.lower().replace({'1.0': 1, '0.0': 0, '1': 1, '0': 0})
    df = df[df['Churn'].isin([0, 1])]
    df['Churn'] = df['Churn'].astype(int)

    # Fix negative values
    df.loc[df['IsActiveMember'] < 0, 'IsActiveMember'] = 0
    df['IsActiveMember'] = df['IsActiveMember'].fillna(0).astype(int)
    df.loc[df['Balance'] < 0, 'Balance'] = df[df['Balance'] >= 0]['Balance'].median()
    df.loc[df['EstimatedSalary'] < 0, 'EstimatedSalary'] = df[df['EstimatedSalary'] >= 0]['EstimatedSalary'].median()
    df = df[df['HasCrCard'].isin([0.0, 1.0])]

    # One-Hot Encoding for categorical features
    df = pd.get_dummies(df, columns=['Gender', 'HasCrCard'], drop_first=False)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Outlier capping
    numeric_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    return df

if __name__ == '__main__':   
    # Load the raw data
    raw_data = pd.read_csv('exl_credit_card_churn_data.csv')

    print("--- Raw Data ---")
    print(raw_data.head())
    
    # Perform  data understanding
    understand_data(raw_data.copy())
    
    #  cleaning and preprocessing
    cleaned_data = clean_data(raw_data.copy())
    
    print("\n--- Cleaned Data ---")
    print(cleaned_data.head())
    
    # Save the cleaned data to a CSV file
    cleaned_data.to_csv('churn_cleaned.csv', index=False)
    print("\nCleaned data saved to 'churn_cleaned.csv'")


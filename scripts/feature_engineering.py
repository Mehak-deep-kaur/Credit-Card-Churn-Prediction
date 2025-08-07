# scripts/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def engineer_and_scale_features(df):
    # Feature Engineering
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['TenurePerProduct'] = df['Tenure'] / (df['NumOfProducts'] + 1)
    balance_75 = df['Balance'].quantile(0.75)
    df['IsHighValueCustomer'] = (df['Balance'] > balance_75).astype(int)

    # Scaling
    numeric_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=[f"{col}_scaled" for col in numeric_cols])

    # Concatenate scaled features with the original dataframe
    df = pd.concat([df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

    # Define final features and target
    features = [
        'Age_scaled', 'Tenure_scaled', 'Balance_scaled', 'NumOfProducts_scaled', 'EstimatedSalary_scaled',
        'IsActiveMember', 'Gender_Female', 'Gender_Male', 'HasCrCard_0.0', 'HasCrCard_1.0',
        'BalanceSalaryRatio', 'TenurePerProduct', 'IsHighValueCustomer'
    ]
    X = df[features]
    y = df['Churn']

    return X, y, scaler

if __name__ == '__main__':
    # This block runs if the script is executed directly
    cleaned_data = pd.read_csv('churn_cleaned.csv')
    X_features, y_target, final_scaler = engineer_and_scale_features(cleaned_data)

    # Save the scaler model to the models folder
    joblib.dump(final_scaler, 'models/scaler_model.pkl')
    print("Scaler model saved to 'models/scaler_model.pkl'")

    # You could also save the final features for later use
    pd.concat([X_features, y_target], axis=1).to_csv('features_and_target.csv', index=False)
    print("Final features and target saved to 'features_and_target.csv'")

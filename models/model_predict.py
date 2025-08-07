# # scripts/model_predict.py

# import pandas as pd
# import joblib
# from data_cleaning_and_preprocessing import clean_data

# def predict_new_customer(raw_data_df):
#     """
#     Loads a trained model and scaler to predict churn for new customer data.

#     Args:
#         raw_data_df (pd.DataFrame): A DataFrame containing new, raw customer data.

#     Returns:
#         list: A list of predictions for each customer in the input DataFrame.
#     """
#     try:
#         # Load the trained model and scaler from the models directory
#         model = joblib.load('models/random_forest.pkl')
#         scaler = joblib.load('models/scaler_model.pkl')
#         print("Successfully loaded the trained model and scaler.")
#     except FileNotFoundError:
#         print("Error: Model or scaler files not found. Please run the main pipeline first.")
#         return None

#     # Apply the same data cleaning and preprocessing steps as the training pipeline
#     cleaned_df = clean_data(raw_data_df.copy())
    
#     # --- FIX: Ensure all required columns are created and in the correct order ---
    
#     # List of all feature columns expected by the model
#     features = [
#         'Age_scaled', 'Tenure_scaled', 'Balance_scaled', 'NumOfProducts_scaled', 'EstimatedSalary_scaled',
#         'IsActiveMember', 'Gender_Female', 'Gender_Male', 'HasCrCard_0.0', 'HasCrCard_1.0',
#         'BalanceSalaryRatio', 'TenurePerProduct', 'IsHighValueCustomer'
#     ]

#     # Initialize a new DataFrame with all required columns and fill with zeros
#     X_predict = pd.DataFrame(0, index=cleaned_df.index, columns=features)
    
#     # Map the columns from the cleaned_df to the new X_predict DataFrame
#     # 1. Map one-hot encoded and binary columns
#     ohe_and_binary_cols = ['IsActiveMember', 'Gender_Female', 'Gender_Male', 'HasCrCard_0.0', 'HasCrCard_1.0']
#     for col in ohe_and_binary_cols:
#         if col in cleaned_df.columns:
#             X_predict[col] = cleaned_df[col]

#     # 2. Scale the numeric features and add to X_predict
#     numeric_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
#     scaled_data = scaler.transform(cleaned_df[numeric_cols])
#     scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_scaled" for col in numeric_cols])
#     for col in scaled_df.columns:
#         X_predict[col] = scaled_df[col]
    
#     # 3. Create and add engineered features
#     X_predict['BalanceSalaryRatio'] = cleaned_df['Balance'] / (cleaned_df['EstimatedSalary'] + 1)
#     X_predict['TenurePerProduct'] = cleaned_df['Tenure'] / (cleaned_df['NumOfProducts'] + 1)
#     balance_75 = pd.Series(cleaned_df['Balance'].quantile(0.75), index=cleaned_df.index)
#     X_predict['IsHighValueCustomer'] = (cleaned_df['Balance'] > balance_75).astype(int)

#     # --- END OF FIX ---

#     # Make predictions
#     predictions = model.predict(X_predict)
#     return predictions

# if __name__ == '__main__':
#     # Example usage with a new, raw customer data point
#     new_customer_data = pd.DataFrame([{
#         'Gender': 'Female',
#         'Age': 45,
#         'Tenure': 5,
#         'Balance': 120000,
#         'NumOfProducts': 1,
#         'IsActiveMember': 1,
#         'EstimatedSalary': 65000,
#         'HasCrCard': 'Yes',
#         'Churn': 0 # Churn is included to match the original data format but will not be used for prediction
#     }])

#     print("New Customer Data:")
#     print(new_customer_data)
    
#     predictions = predict_new_customer(new_customer_data)

#     if predictions is not None:
#         print("\nPrediction for new customer:", predictions[0])
#         print("0 means 'Will Not Churn', 1 means 'Will Churn'")

# scripts/model_predict.py

import pandas as pd
import joblib
from data_cleaning_and_preprocessing import clean_data

def predict_new_customer(raw_data_df):
    """
    Loads a trained model and scaler to predict churn for new customer data.

    Args:
        raw_data_df (pd.DataFrame): A DataFrame containing new, raw customer data.

    Returns:
        list: A list of predictions for each customer in the input DataFrame.
    """
    try:
        # Load the trained model and scaler from the models directory
        model = joblib.load('models/random_forest.pkl')
        scaler = joblib.load('models/scaler_model.pkl')
        print("Successfully loaded the trained model and scaler.")
    except FileNotFoundError:
        print("Error: Model or scaler files not found. Please run the main pipeline first.")
        return None

    # Apply the same data cleaning and preprocessing steps as the training pipeline
    cleaned_df = clean_data(raw_data_df.copy())
    
    # --- FIX: Ensure all required columns are created and in the correct order ---
    
    # List of all feature columns expected by the model
    features = [
        'Age_scaled', 'Tenure_scaled', 'Balance_scaled', 'NumOfProducts_scaled', 'EstimatedSalary_scaled',
        'IsActiveMember', 'Gender_Female', 'Gender_Male', 'HasCrCard_0.0', 'HasCrCard_1.0',
        'BalanceSalaryRatio', 'TenurePerProduct', 'IsHighValueCustomer'
    ]

    # Initialize a new DataFrame with all required columns and fill with zeros
    X_predict = pd.DataFrame(0, index=cleaned_df.index, columns=features)
    
    # Map the columns from the cleaned_df to the new X_predict DataFrame
    # 1. Map one-hot encoded and binary columns
    ohe_and_binary_cols = ['IsActiveMember', 'Gender_Female', 'Gender_Male', 'HasCrCard_0.0', 'HasCrCard_1.0']
    for col in ohe_and_binary_cols:
        if col in cleaned_df.columns:
            X_predict[col] = cleaned_df[col]

    # 2. Scale the numeric features and add to X_predict
    numeric_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaled_data = scaler.transform(cleaned_df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_scaled" for col in numeric_cols])
    for col in scaled_df.columns:
        X_predict[col] = scaled_df[col]
    
    # 3. Create and add engineered features
    X_predict['BalanceSalaryRatio'] = cleaned_df['Balance'] / (cleaned_df['EstimatedSalary'] + 1)
    X_predict['TenurePerProduct'] = cleaned_df['Tenure'] / (cleaned_df['NumOfProducts'] + 1)
    balance_75 = pd.Series(cleaned_df['Balance'].quantile(0.75), index=cleaned_df.index)
    X_predict['IsHighValueCustomer'] = (cleaned_df['Balance'] > balance_75).astype(int)

    # --- END OF FIX ---

    # Make predictions
    predictions = model.predict(X_predict)
    return predictions

if __name__ == '__main__':
    # Example usage with a new, raw customer data point
    new_customer_data = pd.DataFrame([
        {
            'Gender': 'Female',
            'Age': 45,
            'Tenure': 5,
            'Balance': 120000,
            'NumOfProducts': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 65000,
            'HasCrCard': 'Yes',
            'Churn': 0
        },
        {
            'Gender': 'Male',
            'Age': 32,
            'Tenure': 2,
            'Balance': 5000,
            'NumOfProducts': 2,
            'IsActiveMember': 0,
            'EstimatedSalary': 30000,
            'HasCrCard': 'No',
            'Churn': 0
        },
        {
            'Gender': 'Female',
            'Age': 58,
            'Tenure': 10,
            'Balance': 150000,
            'NumOfProducts': 1,
            'IsActiveMember': 0,
            'EstimatedSalary': 110000,
            'HasCrCard': 'Yes',
            'Churn': 0
        },
        {
            'Gender': 'Male',
            'Age': 25,
            'Tenure': 1,
            'Balance': 0,
            'NumOfProducts': 3,
            'IsActiveMember': 1,
            'EstimatedSalary': 25000,
            'HasCrCard': 'No',
            'Churn': 0
        }
    ])

    print("New Customer Data:")
    print(new_customer_data)
    
    predictions = predict_new_customer(new_customer_data)

    if predictions is not None:
        print("\nPredictions for new customers:")
        prediction_labels = {0: 'Non Churn', 1: 'Churn'}
        for i, prediction in enumerate(predictions):
            customer_info = new_customer_data.iloc[i][['Gender', 'Age', 'Balance']]
            print(f"Customer {i+1} (Gender: {customer_info['Gender']}, Age: {customer_info['Age']}, Balance: {customer_info['Balance']}): Prediction - {prediction_labels[prediction]}")



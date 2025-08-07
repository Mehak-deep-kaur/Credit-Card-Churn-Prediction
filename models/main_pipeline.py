import pandas as pd
import os
import sys
import joblib

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import functions from your separate scripts
from data_cleaning_and_preprocessing import clean_data
from feature_engineering import engineer_and_scale_features
from model_training import train_and_evaluate_model
# from EDA_visuals import generate_eda_visuals
from model_predict import predict_new_customer

def run_pipeline():
    print("Starting the Customer Churn Prediction pipeline...")

    # Ensure the models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory.")
    
    # === STEP 1: Data Cleaning and Preprocessing ===
    print("\n--- Step 1: Cleaning and Preprocessing Data ---")
    raw_data_path = 'exl_credit_card_churn_data.csv'
    cleaned_data_path = 'churn_cleaned.csv'

    if os.path.exists(raw_data_path):
        raw_data = pd.read_csv(raw_data_path)
        cleaned_data = clean_data(raw_data)
        cleaned_data.to_csv(cleaned_data_path, index=False)
        print(f"Cleaned data saved to '{cleaned_data_path}'")
    else:
        print(f"Error: Raw data file '{raw_data_path}' not found. Exiting.")
        return

    # === STEP 2: Feature Engineering and Scaling ===
    print("\n--- Step 2: Engineering Features and Scaling Data ---")
    X, y, scaler = engineer_and_scale_features(cleaned_data)
    
    # Save the scaler model
    joblib.dump(scaler, 'models/scaler_model.pkl')
    print("Scaler model saved to 'models/scaler_model.pkl'")

    # Save the final features and target for the next step
    pd.concat([X, y], axis=1).to_csv('features_and_target.csv', index=False)
    print("Final features and target saved to 'features_and_target.csv'")

    # === STEP 3: Model Training and Evaluation ===
    print("\n--- Step 3: Training and Evaluating the Model ---")
    trained_model = train_and_evaluate_model(X, y)

    # Save the final model
    joblib.dump(trained_model, 'models/random_forest.pkl')
    print("Trained model saved to 'models/random_forest.pkl'")

    # # === STEP 4: Generate EDA Visuals ===
    # print("\n--- Step 4: Generating EDA and Visualizations ---")
    # generate_eda_visuals()
    
    # === STEP 5: Make a Prediction ===
    print("\n--- Step 5: Making new predictions ---")
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
            'Gender': 'Female',
            'Age': 60,
            'Tenure': 2,
            'Balance': 100000,
            'NumOfProducts': 1,
            'IsActiveMember': 0,
            'EstimatedSalary': 75000,
            'HasCrCard': 'Yes',
            'Churn': 1  # This customer is likely to churn
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

    print("\nPipeline execution complete.")

if __name__ == '__main__':
    run_pipeline()

# scripts/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def train_and_evaluate_model(X, y):
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    final_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
    )
    final_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = final_model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return final_model

if __name__ == '__main__':
    # This block runs if the script is executed directly
    # Load data
    features_and_target = pd.read_csv('features_and_target.csv')
    X_features = features_and_target.drop('Churn', axis=1)
    y_target = features_and_target['Churn']

    # Train the model
    trained_model = train_and_evaluate_model(X_features, y_target)

    # Save the trained model to the models folder
    joblib.dump(trained_model, 'models/random_forest.pkl')
    print("\nTrained model saved to 'models/random_forest.pkl'")

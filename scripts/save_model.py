import joblib
import pandas as pd

# Import from final_churn.py
from final_churn import final_model, scaler, features

# Save model
joblib.dump(final_model, 'models/random_forest_churn_model.pkl')

# Save scaler
joblib.dump(scaler, 'models/final_model.pkl')

# Save feature list
pd.Series(features).to_csv('models/feature_columns.csv', index=False)

print(" Model, scaler, and features saved successfully to 'models/'")

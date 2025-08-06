import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

# ===== Load Data =====
data = pd.read_csv('exl_credit_card_churn_data.csv')
print("Initial shape:", data.shape)

# ===== Clean & Impute =====
data['Gender'] = data['Gender'].astype(str).str.strip().str.capitalize().replace('Nan', pd.NA)
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])

data['Age'] = data['Age'].apply(lambda x: np.nan if pd.isna(x) or x < 0 or x > 100 else x)
data['Age'] = data['Age'].fillna(data['Age'].median())

for col in ['Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].fillna(data[col].median())

data['HasCrCard'] = data['HasCrCard'].replace({'Yes': 1, 'No': 0, '1': 1, '0': 0})
data['IsActiveMember'] = data['IsActiveMember'].replace({'Yes': 1, 'No': 0, '1': 1, '0': 0})
data['HasCrCard'] = pd.to_numeric(data['HasCrCard'], errors='coerce').fillna(0)
data['IsActiveMember'] = pd.to_numeric(data['IsActiveMember'], errors='coerce').fillna(0)

data['Churn'] = data['Churn'].astype(str).str.strip().str.lower().replace({'1.0': 1, '0.0': 0, '1': 1, '0': 0})
data = data[data['Churn'].isin([0, 1])]
data['Churn'] = data['Churn'].astype(int)

# ===== Fix Negatives =====
data.loc[data['IsActiveMember'] < 0, 'IsActiveMember'] = 0
data['IsActiveMember'] = data['IsActiveMember'].fillna(0).astype(int)
data.loc[data['Balance'] < 0, 'Balance'] = data[data['Balance'] >= 0]['Balance'].median()
data.loc[data['EstimatedSalary'] < 0, 'EstimatedSalary'] = data[data['EstimatedSalary'] >= 0]['EstimatedSalary'].median()
data = data[data['HasCrCard'].isin([0.0, 1.0])]

# ===== One-Hot Encoding =====
data = pd.get_dummies(data, columns=['Gender', 'HasCrCard'], drop_first=False)
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)

# ===== Cap Outliers =====
def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
    return df

numeric_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
for col in numeric_cols:
    data = cap_outliers(data, col)

# ===== Feature Engineering =====
data['BalanceSalaryRatio'] = data['Balance'] / (data['EstimatedSalary'] + 1)
data['TenurePerProduct'] = data['Tenure'] / (data['NumOfProducts'] + 1)
balance_75 = data['Balance'].quantile(0.75)
data['IsHighValueCustomer'] = (data['Balance'] > balance_75).astype(int)

data.to_csv('churn_cleaned.csv', index=False)
print("\n Cleaned data saved to 'churn_cleaned.csv'")
# ===== Scaling =====
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[numeric_cols])
scaled_df = pd.DataFrame(scaled, columns=[f"{col}_scaled" for col in numeric_cols])
data = pd.concat([data.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

# ===== Final Features =====
features = [
    'Age_scaled', 'Tenure_scaled', 'Balance_scaled', 'NumOfProducts_scaled', 'EstimatedSalary_scaled',
    'IsActiveMember', 'Gender_Female', 'Gender_Male', 'HasCrCard_0.0', 'HasCrCard_1.0',
    'BalanceSalaryRatio', 'TenurePerProduct', 'IsHighValueCustomer'
]
X = data[features]
y = data['Churn']

# ===== Train-Test Split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Train Random Forest =====
final_model = RandomForestClassifier(
    n_estimators=1000,           # Increase trees for better generalization
    max_depth=7,                 # Allow deeper trees for more complexity
    min_samples_split=6,         # Prevent splits on small noisy samples
    min_samples_leaf=4,          # Avoid leaves with single outliers
    max_features='sqrt',         # Optimal for RF
    class_weight='balanced',     # Handle class imbalance
    random_state=42,
)

final_model.fit(X_train, y_train)

# ===== Evaluation =====
y_pred = final_model.predict(X_test)
print(f"\n Accuracy:,{ accuracy_score(y_test, y_pred):.2f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===== Feature Importances =====
importances = final_model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
print("\n🔥 Top Feature Importances:\n", feat_imp_df.head(10))

# ===== Visualization =====
plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'][:15], feat_imp_df['Importance'][:15])
plt.gca().invert_yaxis()
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# ===== Save Model =====
joblib.dump(final_model, 'random_forest_churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ===== Class distribution =====
print("\n🎯 Class Distribution:")
print(data['Churn'].value_counts(normalize=True))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===== Load Cleaned Data =====
df = pd.read_csv('churn_cleaned.csv')
print("Data Loaded: ", df.shape)

# ===== Create output directory for plots =====
# output_dir = "feature/eda/plots"
# os.makedirs(output_dir, exist_ok=True)

# ===== Plot 1: Churn Distribution =====
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
# plt.savefig(f"{output_dir}/churn_distribution.png")
plt.show()
plt.close()

# ===== Plot 2: Age Distribution by Churn =====
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Churn', bins=20, kde=True)
plt.title("Age Distribution by Churn")
# plt.savefig(f"{output_dir}/age_distribution.png")
plt.show()
plt.close()

# ===== Plot 3: Balance vs Salary (scatter) =====
plt.figure(figsize=(8, 5))
sns.scatterplot(x='EstimatedSalary', y='Balance', hue='Churn', data=df)
plt.title("Balance vs Estimated Salary by Churn")
# plt.savefig(f"{output_dir}/balance_vs_salary.png")
plt.show()
plt.close()

# ===== Plot 4: Tenure by Churn (Boxplot) =====
plt.figure(figsize=(6, 4))
sns.boxplot(x='Churn', y='Tenure', data=df)
plt.title("Tenure vs Churn")
# plt.savefig(f"{output_dir}/tenure_vs_churn.png")
plt.show()
plt.close()

# ===== Plot 5: Gender vs Churn =====
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender_Male', hue='Churn', data=df)
plt.xticks(ticks=[0, 1], labels=["Female", "Male"])
plt.title("Gender vs Churn")
# plt.savefig(f"{output_dir}/gender_vs_churn.png")
plt.show()
plt.close()

# ===== Plot 6: Active Members vs Churn =====
plt.figure(figsize=(6, 4))
sns.countplot(x='IsActiveMember', hue='Churn', data=df)
plt.title("Active Member vs Churn")
# plt.savefig(f"{output_dir}/active_vs_churn.png")
plt.show()
plt.close()

# ===== Plot 7: Correlation Heatmap =====
plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
# plt.savefig(f"{output_dir}/correlation_matrix.png")
plt.show()
plt.close()

# ===== Plot 8: Boxplots of numeric features =====
num_cols = ['Age', 'Tenure', 'Balance', 'EstimatedSalary', 'NumOfProducts']
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f"{col} vs Churn")
    # plt.savefig(f"{output_dir}/boxplot_{col}.png")
    plt.show()
    plt.close()

# print(f"\n✅ All EDA plots saved to: {output_dir}")



# ===== Load Cleaned Data =====
df = pd.read_csv('churn_cleaned.csv')
print("Data Loaded: ", df.shape)
data = pd.read_csv('exl_credit_card_churn_data.csv')

# ===== Color Palette for Churn =====
churn_colors = ["#4CAF50", "#F44336"]  # Green for No churn, Red for Churn

# ===== Plot Group 1: Churn Overview =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(ax=axes[0], x='Churn', data=df, palette=churn_colors)
axes[0].set_title("Churn Distribution")
axes[0].set_xlabel("Churn (0 = No, 1 = Yes)")

sns.countplot(ax=axes[1], x='Gender_Male', hue='Churn', data=df, palette=churn_colors)
axes[1].set_title("Gender vs Churn")
axes[1].set_xlabel("Gender (0 = Female, 1 = Male)")

sns.countplot(ax=axes[2], x='IsActiveMember', hue='Churn', data=df, palette=churn_colors)
axes[2].set_title("Active Member vs Churn")
axes[2].set_xlabel("Is Active Member")

plt.tight_layout()
plt.show()

# ===== Plot Group 2: Age & Tenure =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(ax=axes[0], data=df, x='Age', hue='Churn', bins=20, kde=True, palette=churn_colors)
axes[0].set_title("Age Distribution by Churn")

sns.boxplot(ax=axes[1], x='Churn', y='Tenure', data=df, palette=churn_colors)
axes[1].set_title("Tenure vs Churn")

plt.tight_layout()
plt.show()

# ===== Plot Group 3: Financial Features =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(ax=axes[0], x='EstimatedSalary', y='Balance', hue='Churn', data=df, palette=churn_colors)
axes[0].set_title("Balance vs Estimated Salary by Churn")

sns.boxplot(ax=axes[1], x='Churn', y='Balance', data=df, palette=churn_colors)
axes[1].set_title("Balance vs Churn")

plt.tight_layout()
plt.show()

# ===== Plot Group 4: More Numerical Features =====
num_cols = ['Age', 'Tenure', 'Balance', 'EstimatedSalary', 'NumOfProducts']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(num_cols):
    sns.boxplot(ax=axes[idx], x='Churn', y=col, data=df, palette=churn_colors)
    axes[idx].set_title(f"{col} vs Churn")

# Hide extra subplot if number is odd
if len(num_cols) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
plt.show()

# ===== Plot: Correlation Heatmap =====
plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(data.isnull(), cbar=False, cmap='mako')
plt.title("Missing Values - Cleaned Data")
plt.show()
# ===== Plot: Missing Values Heatmap (optional - data is already clean) =====
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='mako')
plt.title("Missing Values - Cleaned Data")
plt.show()

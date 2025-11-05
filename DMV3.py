import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Telecom_Customer_Churn.csv")
print(" Dataset Loaded\n")

print(" First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\n Summary Statistics:")
print(df.describe(include='all'))

# 3. Handle Missing Values
df.replace(" ", np.nan, inplace=True)   # Replace empty strings with NaN
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric missing with median
df.fillna(df.mode().iloc[0], inplace=True)             # Fill categorical missing with mode

print("\nMissing values handled")

# 4. Remove Duplicates
df.drop_duplicates(inplace=True)
print(" Duplicates removed")

# 5. Fix inconsistent formatting
df.columns = df.columns.str.strip()   # Remove spaces in column names
df['gender'] = df['gender'].str.title()  # Example: male->Male, FEMALE->Female

if "Yes" in df["Churn"].unique():
    df['Churn'] = df['Churn'].str.strip().str.title()  # yes/YES → Yes

print("Inconsistent text formatting standardized")

# 6. Convert Data Types
numeric_cols = ["TotalCharges", "MonthlyCharges", "tenure"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(" Data types corrected")

# 7. Handle Outliers → Using IQR Method
for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

print("Outliers handled")

# 8. Feature Engineering
df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)  # Average per month
df["IsSeniorCitizen"] = df["SeniorCitizen"].apply(lambda x: 1 if x == 1 else 0)

# 9. Scale / Normalize Numerical Features
scaler = StandardScaler()
num_features = ["MonthlyCharges", "TotalCharges", "tenure", "AvgCharges"]

df[num_features] = scaler.fit_transform(df[num_features])
print(" Data scaled")

# 10. Train-Test Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train-Test split completed")

# 11. Export Cleaned Dataset
df.to_csv("Telecom_Customer_Churn_Cleaned.csv", index=False)
print("Cleaned dataset saved as 'Telecom_Customer_Churn_Cleaned.csv'")

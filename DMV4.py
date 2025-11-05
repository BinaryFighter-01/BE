import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("Bengaluru_House_Data.csv")
print("Dataset loaded")

# Clean column names
df.columns = df.columns.str.lower().str.replace(" ", "_")
print("Column names cleaned")

# 2. Handle missing values
df.replace(" ", np.nan, inplace=True)

# Fill numeric missing values with median
num_cols = df.select_dtypes(include=['float64','int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical missing values with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print(" Missing values handled")

# 3. Feature Engineering

# Convert size column: "2 BHK" → 2
df["bhk"] = df["size"].apply(lambda x: int(str(x).split()[0]))

# Convert sqft column
def convert_sqft(x):
    try:
        if '-' in x:
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return np.nan

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
# df["total_sqft"].fillna(df["total_sqft"].median(), inplace=True) # ithe warning yete


df.fillna({"total_sqft":df["total_sqft"].median()},inplace=True) #to avoid warning

# Price per sqft
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

print("Feature engineering completed")

# 4. Remove duplicates
df.drop_duplicates(inplace=True)

# 5. Filter / Subset: Only reasonable totals
df = df[df["total_sqft"] / df["bhk"] >= 300]   # remove unrealistic sqft/BHK

# 6. Handle outliers using IQR
for col in ["price", "total_sqft", "bath"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

print(" Outliers handled")

# 7. Encode categorical variables
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["location"] = encoder.fit_transform(df["location"])
df["area_type"] = encoder.fit_transform(df["area_type"])
df["availability"] = encoder.fit_transform(df["availability"])

print(" Encoding done")

# 8. Group Summary - Avg price per location
summary = df.groupby("location")["price"].mean().reset_index(name="avg_price")
print("\n Sample Summary (Avg price per location):")
print(summary.head())

# 9. Export Cleaned Data
df.to_csv("Bengaluru_House_Data_Cleaned.csv", index=False)
print(" Clean file saved as Bengaluru_House_Data_Cleaned.csv")

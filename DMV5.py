# --- Import libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#PREPROCESSING 

# Load dataset
df = pd.read_csv("air_quality.csv")   #  Replace with your actual filename

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])  # Remove rows with invalid dates

# Convert numeric columns safely
cols = ["PM2.5", "PM10", "CO", "AQI"]
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

#Fill missing values with column medians
df[cols] = df[cols].fillna(df[cols].median())

# Remove duplicates and invalid (negative) values
df = df.drop_duplicates()
for c in cols:
    df = df[df[c] >= 0]

#  Remove outliers using IQR method
for c in cols:
    Q1, Q3 = df[c].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df[c] >= Q1 - 1.5 * IQR) & (df[c] <= Q3 + 1.5 * IQR)]

#  Resample to monthly averages
df.set_index("Date", inplace=True)
df_month = df.resample("M").mean(numeric_only=True)

print("Data Preprocessing Done!\n")
print(df_month.head())

# ----------------- VISUALIZATION -----------------

# 1. AQI Time Series Line Plot
plt.figure(figsize=(8,4))
plt.plot(df_month.index, df_month["AQI"], marker='o', color='red')
plt.title("Monthly AQI Trend")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2 Pollutants Trend Plot
plt.figure(figsize=(8,4))
plt.plot(df_month.index, df_month["PM2.5"], label="PM2.5")
plt.plot(df_month.index, df_month["PM10"], label="PM10")
plt.plot(df_month.index, df_month["CO"], label="CO")
plt.title("Monthly Pollutant Levels")
plt.xlabel("Date")
plt.ylabel("Pollution Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#  3. Bar Plot — Top 20 Highest AQI Days
top20 = df.nlargest(20, "AQI")
plt.figure(figsize=(9,4))
plt.bar(top20.index.strftime("%Y-%m-%d"), top20["AQI"], color='orange')
plt.title("Top 20 Highest AQI Days")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 4. Box Plot — AQI Distribution
plt.figure(figsize=(5,4))
plt.boxplot(df["AQI"], patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title("AQI Distribution")
plt.ylabel("AQI")
plt.tight_layout()
plt.show()

# 5. Scatter Plot — AQI vs PM2.5
plt.figure(figsize=(5,4))
plt.scatter(df["PM2.5"], df["AQI"], alpha=0.5, color='purple')
plt.title("AQI vs PM2.5")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.grid(True)
plt.tight_layout()
plt.show()

print("All Plots Generated Successfully!")

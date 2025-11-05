# Simple Uber Fare Prediction using Regression Models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv("uber.csv")

# Basic preprocessing
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.dayofweek
df.drop(['key', 'pickup_datetime', 'Unnamed: 0'], axis=1, inplace=True, errors='ignore')
df.fillna(df.mean(), inplace=True)

# Remove outliers (IQR method)
Q1, Q3 = df['fare_amount'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df['fare_amount'] >= Q1 - 1.5*IQR) & (df['fare_amount'] <= Q3 + 1.5*IQR)]

#  Correlation check
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Split data
X = df.drop('fare_amount', axis=1)
y = df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# Train & evaluate models
models = {"Linear": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso()}
print("\nModel Performance:\n-----------------------")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name:6s} | R²={r2_score(y_test, pred):.3f} | RMSE={np.sqrt(mean_squared_error(y_test, pred)):.2f}")

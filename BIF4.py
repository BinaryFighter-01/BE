# Machine Learning for Genomic Data Classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Simulated genomic dataset
np.random.seed(42)
X = np.random.rand(100, 10) * 100
y = np.random.choice([0, 1], size=100)

df = pd.DataFrame(X, columns=[f'Gene_{i+1}' for i in range(10)])
df['Label'] = y

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Label', axis=1),
    df['Label'],
    test_size=0.3,
    random_state=42
)

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate models
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion matrices
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Feature importance (Random Forest)
importances = rf_model.feature_importances_
features = X_train.columns

plt.barh(features, importances, color='skyblue')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Genomic Features")
plt.show()

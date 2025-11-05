# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load the dataset
df = pd.read_csv('Wine.csv') 

# Step 3: Split into features (X) and target (y)
X = df.drop('Customer_Segment', axis=1)  # Independent variables
y = df['Customer_Segment']               # Target class (wine type)

# Step 4: Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA (reduce to 2 principal components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 6: Show how much variance each principal component explains
print("\nVariance explained by components:")
print("PC1:", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("PC2:", round(pca.explained_variance_ratio_[1] * 100, 2), "%")

# Step 7: Plot the PCA-transformed data
plt.figure(figsize=(8, 6))

# Loop through each wine type and plot separately
for wine_type in [1, 2, 3]:
    mask = (y == wine_type)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=f'Wine Type {wine_type}', s=50)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Wine Classification using PCA')
plt.legend()
plt.grid(True)
plt.show()


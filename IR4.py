# Agglomerative Hierarchical Clustering Example

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Iris dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Standardize features
X = StandardScaler().fit_transform(X)

# Create linkage matrix for dendrogram
linkage_matrix = linkage(X, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Euclidean Distance")
plt.show()

# Apply Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)

# Display results
print("Cluster labels for each data point:")
print(labels)

# # Optional: visualize clusters using first two features
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
# plt.title("Agglomerative Clustering Results (Iris Data)")
# plt.xlabel(feature_names[0])
# plt.ylabel(feature_names[1])
# plt.show()

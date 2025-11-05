import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt

digits = load_digits()
X, Y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Improved model — better kernel and tuned hyperparameters
model = svm.SVC(kernel='rbf', gamma=0.001, C=10)
model.fit(X_train, y_train)

n = 5
predicted = model.predict([digits.data[n]])

plt.imshow(digits.images[n], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title(f"Predicted Digit: {predicted[0]}")
plt.axis("off")
plt.show()

print(f"Predicted Value: {predicted[0]}")
print(f"Actual Value: {digits.target[n]}")

# Optional: check accuracy
print(f"\nModel Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

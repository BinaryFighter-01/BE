import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('car_evaluation.csv')
print("File read successfully!")
print(data.head())

data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

data.head()

encoder = LabelEncoder()
for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print(" Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sample_input = pd.DataFrame({
    'buying': [2],    # encoded values
    'maint': [1],
    'doors': [2],
    'persons': [3],
    'lug_boot': [1],
    'safety': [2]
})
pred = rf_model.predict(sample_input)
print("\nPredicted Safety Class for sample car:", pred[0])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Training data (emails + labels)
emails = [
     "Hi I wanted to meet you",
     "Lowest price for your medicines",
     "Hi Anil win exciting prizes",
     "Congratulations! You are a certified OCI developer",
     "Get your rewards now",
     "Limited offer buy 1 get 1 free!",
     "Earn dollars in fast way",
     "Today is the final day of project submission",
     "Yes! You are eligible to get loan"
]

labels = ["ham", "spam", "spam", "ham", "spam", "spam", "ham", "ham", "spam"]

# Step 2: Convert text into numerical features (Bag of Words)
vec = CountVectorizer()
X = vec.fit_transform(emails)

# Step 3: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

# Step 4: Take user input and predict
test = input("Enter a test message: ") 
pred = model.predict(vec.transform([test]))  
print("\nPrediction:", pred[0])


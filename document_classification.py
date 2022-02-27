# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
data = pd.read_csv("documents.csv")

# Split data into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Vectorize the text data
vectorizer = CountVectorizer()
train_X = vectorizer.fit_transform(train_data['content'])
test_X = vectorizer.transform(test_data['content'])

# Train the model
model = MultinomialNB()
model.fit(train_X, train_data['category'])

# Test the model
accuracy = model.score(test_X, test_data['category'])
print("Accuracy:", accuracy)

# Classify new documents
new_documents = ["This is a document about sports.", "A new recipe for a dessert."]
new_X = vectorizer.transform(new_documents)
predictions = model.predict(new_X)
print("Predictions:", predictions)
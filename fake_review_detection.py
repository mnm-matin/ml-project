python
# Import necessary libraries for the ML project
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
df = pd.read_csv('reviews.csv')
df['label'] = np.where(df['rating'] >= 4, 'Genuine', 'Fake')
X = df['review_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a logistic regression model on the vectorized data
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model on the test data and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy of the model
print('Accuracy of the model:', accuracy)

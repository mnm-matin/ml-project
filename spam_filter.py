# Import required libraries
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load data into a pandas dataframe
data = pd.read_csv('spam.csv', encoding='latin-1')

# Split data into training and testing sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# Extract email contents and labels
train_emails = train_data['v2']
train_labels = train_data['v1']
test_emails = test_data['v2']
test_labels = test_data['v1']

# Create bag of words representation of email contents
vectorizer = CountVectorizer(stop_words='english')
train_features = vectorizer.fit_transform(train_emails)
test_features = vectorizer.transform(test_emails)

# Train a Naive Bayes classifier on training data
classifier = MultinomialNB()
classifier.fit(train_features, train_labels)

# Generate predictions on test data
predictions = classifier.predict(test_features)

# Evaluate performance of the classifier using accuracy score, precision, recall and f1-score
accuracy = metrics.accuracy_score(test_labels, predictions)
precision = metrics.precision_score(test_labels, predictions, pos_label='spam')
recall = metrics.recall_score(test_labels, predictions, pos_label='spam')
f1_score = metrics.f1_score(test_labels, predictions, pos_label='spam')

# Print results
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1_score)
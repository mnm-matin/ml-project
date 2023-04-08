# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the data 
data = pd.read_csv('food_items.csv')

# Create feature matrix and target variable
X = data.food_item
y = data.category

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Convert the text data into vectors using CountVectorizer
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# Train the model and make predictions on the test set
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred = nb.predict(X_test_dtm)

# Evaluate the performance of the model
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

# Save the model as a pickle file for future use
import pickle
with open('food_classifier.pkl', 'wb') as f:
    pickle.dump(nb, f)
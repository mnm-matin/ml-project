python
# Import the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('path/to/dataset.csv')

# Vectorize the text data using count vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize a logistic regression model
model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Print the accuracy score of the model
print('Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred)*100))

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reading data from CSV file
data = pd.read_csv('advertisement_clicks.csv')

# Removing unnecessary columns
data = data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1)

# Splitting data into training and testing sets
X = data.drop(['Clicked on Ad'], axis=1)
y = data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting on test data
y_pred = model.predict(X_test)

# Calculating accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)

# Using the model to predict for a given user
new_user = [[25, 'Male', 50000, 3, 2]]
new_user_pred = model.predict(new_user)
print("Prediction for the new user:", new_user_pred)
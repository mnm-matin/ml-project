# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset
data = pd.read_csv('customer_data.csv')

# Splitting the dataset into features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling the features using StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Printing the accuracy of the model
print("Accuracy of the model:", accuracy)

# Predicting whether a new customer will leave or not
new_customer_data = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
new_customer_data[:, 1] = pd.Categorical(new_customer_data[:, 1], categories=data['Geography'].unique()).codes
new_customer_data[:, 2] = pd.Categorical(new_customer_data[:, 2], categories=data['Gender'].unique()).codes
new_customer_data = new_customer_data.astype(float)
new_customer_data = sc.transform(new_customer_data)
new_customer_pred = classifier.predict(new_customer_data)
print("Prediction for the new customer:", new_customer_pred)
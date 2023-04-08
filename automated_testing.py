# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Loading the dataset
data = pd.read_csv('software_testing_data.csv')

# Removing any rows with missing values
data.dropna(inplace=True)

# Splitting the dataset into training and testing sets
X = data.drop('defect', axis=1)
y = data['defect']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the decision tree classifier model
dt = DecisionTreeClassifier(random_state=42)

# Fitting the model with training data
dt.fit(X_train, y_train)

# Predicting the testing set results
y_pred = dt.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print(f"The accuracy of the machine learning model to identify defects in software applications is {accuracy*100:.2f}%")
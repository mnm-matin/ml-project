python
# Importing required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Reading the dataset into a dataframe
df = pd.read_csv("fraud_dataset.csv")

# Splitting the dataset into training and testing sets
x = df.drop(['IsFraud'], axis=1)
y = df['IsFraud']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Creating a Decision Tree model and fitting it to the training data
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train, y_train)

# Predicting labels for the testing data using the Decision Tree model
y_pred = dt_classifier.predict(x_test)

# Calculating the accuracy of the Decision Tree model
dt_accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Classifier Accuracy:", dt_accuracy)

# Creating a Random Forest model and fitting it to the training data
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(x_train, y_train)

# Predicting labels for the testing data using the Random Forest model
y_pred = rf_classifier.predict(x_test)

# Calculating the accuracy of the Random Forest model
rf_accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Classifier Accuracy:", rf_accuracy)

# Comparing the accuracies of both models
if dt_accuracy > rf_accuracy:
    print("Decision Tree Classifier outperforms Random Forest Classifier")
else:
    print("Random Forest Classifier outperforms Decision Tree Classifier")

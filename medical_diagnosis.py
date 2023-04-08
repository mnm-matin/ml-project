# This code imports necessary libraries for machine learning project
import pandas as pd
import numpy as np
import sklearn

# This code imports the dataset CSV file for the project
data = pd.read_csv('medical_conditions.csv')

# This code prepares data for the machine learning model
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# This code splits the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# This code trains the machine learning model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

# This code evaluates the performance of the machine learning model
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) # prints confusion matrix

# This code saves the trained machine learning model for future use
import joblib
joblib.dump(classifier, 'medical_condition_diagnosis_model.joblib')
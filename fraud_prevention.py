
# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('fraudulent_data.csv')

# Divide the data into features and target variable
X = data.drop('fraud', axis=1)
y = data['fraud']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the Random Forest Classifier model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict the target variable
y_pred = model.predict(X_test)

# Find the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Evaluate the model
print('Accuracy of model:', accuracy)

# Deploy the model to detect and prevent fraudulent activities in a system

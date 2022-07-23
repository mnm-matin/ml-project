# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('user_behavior_data.csv')

# Split data into training and test sets
train_data, test_data, train_targets, test_targets = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)

# Preprocess data (fill missing values, scale features, etc.)

# Instantiate Random Forest classifier
rfc = RandomForestClassifier()

# Train classifier
rfc.fit(train_data, train_targets)

# Make predictions on test set
predictions = rfc.predict(test_data)

# Evaluate model accuracy
accuracy = accuracy_score(test_targets, predictions)
print('Model accuracy:', accuracy)
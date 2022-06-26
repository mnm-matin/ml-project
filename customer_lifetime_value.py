# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('customer_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['age', 'gender', 'income']], data['lifetime_value'], test_size=0.2, random_state=42)

# Create a Linear Regression model and fit it to the data
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model's accuracy
print(f'Training Accuracy: {model.score(X_train, y_train)}')
print(f'Testing Accuracy: {model.score(X_test, y_test)}')

# Use the model to predict the lifetime value of a new customer
new_customer = np.array([[35, 'male', 75000]]) # Age, Gender and Income
if new_customer[0,1] == 'male':
  new_customer[0,1] = 1
else:
  new_customer[0,1] = 0
prediction = model.predict(new_customer)
print(f'Predicted Lifetime Value: {prediction}')
# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Loading the data
data = pd.read_csv('sales_data.csv')

#Preparing the data for training
X = data.iloc[:,:-1].values
y = data.iloc[:,1].values

# Splitting the data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating the machine learning model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Making the prediction
y_pred = regressor.predict(X_test)

# Visualizing the results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Demand Forecasting (Training set)')
plt.xlabel('Time (months)')
plt.ylabel('Sales')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Demand Forecasting (Test set)')
plt.xlabel('Time (months)')
plt.ylabel('Sales')
plt.show()
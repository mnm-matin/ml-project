# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Reading data from csv file
data = pd.read_csv('market_data.csv')

# Splitting independent and dependent variables
X = data.iloc[:, :4].values
y = data.iloc[:, -1].values

# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model using linear regression algorithm
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting prices of products or services using trained model
y_pred = regressor.predict(X_test)

# Evaluating performance of model using Mean Squared Error metric
mse = np.mean((y_test - y_pred)**2)
print("Mean Squared Error: ", mse)

# Predicting prices based on market trends and customer behavior
new_data = np.array([[0.5, 0.6, 0.7, 0.8]])
price = regressor.predict(new_data)
print("Predicted Price: ", price)
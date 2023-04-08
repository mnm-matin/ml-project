# Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Reading in dataset
weather_data = pd.read_csv('weather_data.csv')

# Cleaning/preprocessing data 
# (remove unnecessary columns, fill in missing values, etc.)

# Feature engineering
# (creating new features or transforming existing ones to better represent information)

# Splitting data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on testing set
y_pred = model.predict(X_test)

# Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Visualizing model output and data
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
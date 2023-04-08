# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from PIL import Image
import numpy as np
import pandas as pd

# Load and preprocess satellite imagery data
image = Image.open('berlin_satellite_image.jpeg')
data = np.asarray(image)
flat_data = data.flatten()
df = pd.DataFrame({'Pixel_Value': flat_data})

# Load energy demand data
energy_demand = pd.read_csv('energy_demand_data.csv')
df['Energy_Demand'] = energy_demand['Energy_Demand']

# Splitting data into training and testing sets
train, test = train_test_split(df, test_size=0.3, random_state=42)

# Creating an instance of the Random Forest Regressor and training it on the training set
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train[['Pixel_Value']], train['Energy_Demand'])

# Testing the trained model on the testing set
y_pred = rf.predict(test[['Pixel_Value']])
r2 = r2_score(test['Energy_Demand'], y_pred)

# Printing the performance of the model
print("R squared score on the testing set:", r2)
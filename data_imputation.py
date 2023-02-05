# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Reading the dataset 
df = pd.read_csv("dataset.csv")

# Splitting the dataset into train and test set
train, test = train_test_split(df, test_size=0.2, random_state=1)

# Separating the target variable from the independent variables
target = train['target_variable']
train = train.drop('target_variable', axis=1)

# Filling missing values using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
train = imputer.fit_transform(train)

# Training the model using Linear Regression
model = LinearRegression()
model.fit(train, target)

# Filling the missing values in the test set
test = imputer.transform(test.drop('target_variable', axis=1))
test['target_variable'] = model.predict(test)

# Printing the updated dataset
print(test)
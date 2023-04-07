
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('edinburgh_real_estate.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Price', axis=1), df['Price'], test_size=0.2)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate the model's accuracy on the testing data
accuracy = model.score(X_test, y_test)

# Test the model with a new data point
new_data = [[2000, 3, 2, 1, 1]]  # Size(sqft), Bedrooms, Bathrooms, Parking, Year
prediction = model.predict(new_data)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Predicted price: {prediction}")

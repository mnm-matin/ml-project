
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the drug discovery dataset
drugs_df = pd.read_csv('drug_discovery_dataset.csv')

# Preprocess the data
X = drugs_df.drop(['efficacy'], axis=1)
y = drugs_df['efficacy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model's performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f'Training score: {train_score}')
print(f'Testing score: {test_score}')

# Use the trained model to make predictions on new data
new_drug = pd.DataFrame({'property1': [1.0], 'property2': [2.5], 'property3': [0.8]})
prediction = model.predict(new_drug)[0]
print(f'Predicted efficacy of new drug: {prediction}')

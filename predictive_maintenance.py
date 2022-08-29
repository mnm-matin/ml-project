python
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('equipment_usage_data.csv')

# Preprocess data
X = df.drop(['maintenance_needed'], axis=1)
y = df['maintenance_needed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate model
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)

# Make predictions on new data
new_data = pd.read_csv('new_equipment_usage_data.csv')
predictions = clf.predict(new_data)
print('Predictions:', predictions)

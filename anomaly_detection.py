
# Load the dataset
import pandas as pd
dataset = pd.read_csv('my_dataset.csv')

# Preprocess the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(dataset.drop(columns=['anomaly']))
y = dataset['anomaly']

# Train the isolation forest model
from sklearn.ensemble import IsolationForest
model = IsolationForest()
model.fit(X)

# Predict the anomalies
y_pred = model.predict(X)

# Print the anomalies
anomalies = dataset[y_pred == -1]
print(anomalies)

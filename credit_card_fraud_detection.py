python
# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('credit_card_transactions.csv')

# Split data into features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the feature set
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Deal with Class imbalance - Oversampling using SMOTE
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Fit model on the resampled data
model.fit(X_resampled, y_resampled)

# Predict using test data
y_pred = model.predict(X_test)

# Print the confusion matrix and classification report for model evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

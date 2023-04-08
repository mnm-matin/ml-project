
# Import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('creditcard.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.ix[:, data.columns != 'Class'], 
                                                    data['Class'], 
                                                    test_size=0.3,
                                                    random_state=0)

# Train the model using SVM algorithm
clf = SVC(kernel='linear', class_weight='balanced', probability=True)
clf.fit(X_train, y_train)

# Predict the class labels for the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

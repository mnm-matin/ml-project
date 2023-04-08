# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Read in the loan dataset
df = pd.read_csv('loan_data.csv')

# Define the feature and target variables
X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier and fit it to the training data
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = dt_clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print("The accuracy of the model is:", accuracy)
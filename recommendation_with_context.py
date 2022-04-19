# This is an example of how one can create a simple machine learning project using Decision Trees.

from sklearn import tree

# Training data with features and labels
# Features are location and time
# Labels are the products or services that the user is likely to need
X = [[0, 9], [1, 5], [2, 6], [3, 8], [4, 7], [5, 10], [6, 11], [7, 12]]
Y = ["Pizza", "Coffee", "Groceries", "Gas", "Clothing", "Burgers", "Ice Cream", "Electronics"]

# Initializing a Decision Tree Classifier model
clf = tree.DecisionTreeClassifier()

# Fitting the model with the training data
clf = clf.fit(X, Y)

# A sample input from the user
location = 2
time = 9

# Predicting the service that the user is likely to need
prediction = clf.predict([[location, time]])

# Printing the prediction
print("You might like to try: " + prediction[0])
python
#Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#Load dataset into a pandas dataframe
df = pd.read_csv('customer_feedback.csv')

#Create feature matrix and label vector using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(df['feedback']).toarray()
y = df['satisfaction_level'].values

#Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Naive Bayes classifier on the training set
clf = MultinomialNB()
clf.fit(X_train, y_train)

#Predict the satisfaction level of test data
y_pred = clf.predict(X_test)

#Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

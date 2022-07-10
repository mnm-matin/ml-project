
#Import necessary libraries and packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Load dataset
df = pd.read_csv('customer_reviews.csv')

#Create feature and target variables
X = df['review']
y = df['sentiment']

#Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Create a CountVectorizer object to count the frequency of the words in the reviews
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

#Create a Multinomial Naive Bayes classifier and train it
nb = MultinomialNB()
nb.fit(X_train_cv, y_train)

#Test the model by predicting the sentiment of the testing data
y_pred = nb.predict(X_test_cv)

#Print the accuracy score of the model
print('Accuracy score:', nb.score(X_test_cv, y_test))

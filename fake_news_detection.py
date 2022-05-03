'''
The following code creates a Python machine learning project that is able to differentiate between fake news and real news articles.

Before getting started, please make sure you install the necessary libraries required for this project.

'''

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Load the data into a pandas dataframe
df=pd.read_csv('news.csv')

# Drop any null values
df=df.dropna()

# Split the data into train and test sets
x_train,x_test,y_train,y_test=train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)

# Initialize the TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the train set, transform the test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# Initialize the PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Create confusion matrix to evaluate the model
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

# Visualize confusion matrix
cm = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

# Test with some own news
input_text = ['We will have a manned mission to Mars within the next decade.']
tfidf_input=tfidf_vectorizer.transform(input_text)
y_pred_input=pac.predict(tfidf_input)

print(y_pred_input)

'''
The above code can be run from the command line using the following command:

python ml_project.py

Note that the data file 'news.csv' should be present in the same directory as the Python script.
'''
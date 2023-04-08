# This is a python code for a machine learning project to identify the speaker from their voice.

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Load the dataset
dataset = pd.read_csv('voice.csv')

# Convert categorical variables to numerical variables
dataset['label'] = np.where(dataset['label']=='male',0,1)

# Split the dataset into training and testing datasets
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Standardize the independent variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the random forest classifier
classifier = RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# Predict the test results
y_pred = classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()
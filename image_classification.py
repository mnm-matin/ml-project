# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
import os

# Defining the image directory and categories
CATEGORIES = ["animals", "landscapes"]
DIR = "image_dir"

# Extracting the images and putting into lists
images = []
labels = []
for category in CATEGORIES:
    path = os.path.join(DIR, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (50,50))
        images.append(image)
        labels.append(CATEGORIES.index(category))

# Converting the images and labels into arrays
images = np.array(images)
labels = np.array(labels)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Initializing the SVM model and fitting the training data
model = SVC(kernel="linear", C=10)
model.fit(X_train.reshape(len(X_train), -1), y_train)

# Making predictions on the testing data
y_pred = model.predict(X_test.reshape(len(X_test), -1))

# Calculating the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Saving the trained model
import joblib
joblib.dump(model, "image_classifier.pkl")
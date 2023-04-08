# Importing Required Libraries
import numpy as np
import cv2
import os

# Loading the Haar Cascade Face Classifier XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Creating a Function to Detect Faces in an Image and Extracting the Faces
def detect_faces(image):
    # Converting Image to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecting Faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    # Extracting the Faces
    face_images = []
    for (x, y, w, h) in faces:
        face_images.append(gray[y:y + h, x:x + w])

    # Returning the Extracted Faces
    return face_images

# Creating Function to Train our Classifier
def train_face_recognizer(images_folder_path):
    # Initializing Lists to Store Faces and Labels
    faces = []
    labels = []

    # Looping through each Sub-folder in the Images Folder
    for sub_folder_name in os.listdir(images_folder_path):
        sub_folder_path = os.path.join(images_folder_path, sub_folder_name)

        # Looping through each Image in the Sub-folder
        for image_name in os.listdir(sub_folder_path):
            image_path = os.path.join(sub_folder_path, image_name)

            # Loading the Image
            image = cv2.imread(image_path)

            # Detecting Faces in the Image
            face_images = detect_faces(image)

            # Adding the Extracted Faces and Corresponding Label to the Lists
            for face_image in face_images:
                faces.append(face_image)
                labels.append(sub_folder_name)

    # Creating our EigenFaces Recognizer
    face_recognizer = cv2.face.EigenFaceRecognizer_create()

    # Training the Recognizer on the Extracted Faces and Corresponding Labels
    face_recognizer.train(faces, np.array(labels))

    # Returning our Trained Recognizer
    return face_recognizer

# Creating a Function to Predict the Identities of Faces in an Image
def predict_faces(image_path, face_recognizer):
    # Loading the Test Image
    image = cv2.imread(image_path)

    # Detecting Faces in the Test Image
    face_images = detect_faces(image)

    # Predicting the Identities of the Faces
    predictions = []
    for face_image in face_images:
        label, confidence = face_recognizer.predict(face_image)
        predictions.append({'label': label, 'confidence': confidence})

    # Returning the Predictions
    return predictions

# Defining the Path to the Images and Training our Recognizer
images_folder_path = 'images'
face_recognizer = train_face_recognizer(images_folder_path)

# Predicting the Identities of Faces in Test Images
test_image_path = 'test_image.jpg'
predictions = predict_faces(test_image_path, face_recognizer)

# Printing the Predictions
for prediction in predictions:
    print(f"Label: {prediction['label']}, Confidence: {prediction['confidence']}")
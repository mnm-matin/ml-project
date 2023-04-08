# Import necessary libraries
import cv2
import numpy as np

# Load the input images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Detect and extract the faces from the input images
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For image1
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
faces1 = face_detector.detectMultiScale(gray_image1, 1.3, 5)

# For image2
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
faces2 = face_detector.detectMultiScale(gray_image2, 1.3, 5)

# Swap the faces
for (x1,y1,w1,h1) in faces1:
	# Resize the face to fit the other image
	face1 = cv2.resize(image1[y1:y1+h1, x1:x1+w1], (w2,h2))
	
	for (x2,y2,w2,h2) in faces2:
		# Resize the face to fit the other image
		face2 = cv2.resize(image2[y2:y2+h2, x2:x2+w2], (w1,h1))
		
		# Swap the faces
		image1[y1:y1+h1, x1:x1+w1] = face2
		image2[y2:y2+h2, x2:x2+w2] = face1

# Show the output images
cv2.imshow('Swapped Image 1', image1)
cv2.imshow('Swapped Image 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Import OpenCV library
import cv2

# Load pretrained face detection model
face_detection_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load image to be processed
input_image = cv2.imread('input_image.jpg')

# Convert image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image using the face detection model
faces = face_detection_model.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Draw a rectangle around each detected face in the original image
for (x, y, width, height) in faces:
    cv2.rectangle(input_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Display the image with faces detected
cv2.imshow('Faces Detected', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

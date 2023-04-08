# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Load the satellite image
image = cv2.imread('satellite_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Apply Canny edge detection algorithm to the blurred image
edges = cv2.Canny(blur, 50, 150)

# Apply binary thresholding to the edges
ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

# Apply morphological operations to the thresholded image
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Find contours in the opening
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(image, contours, -1, (0,255,0), 2)

# Display the original image with contours
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
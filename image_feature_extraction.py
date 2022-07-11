
import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread('example_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection using Canny algorithm
edges = cv2.Canny(gray, 100, 200)

# Apply Hough Line Transform to extract lines from the image
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Compute the average length and angle of the lines
line_lengths = []
line_angles = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angle = np.arctan2(y2 - y1, x2 - x1)
    line_lengths.append(length)
    line_angles.append(angle)

avg_length = np.mean(line_lengths)
avg_angle = np.mean(line_angles)

# Extract other features from the image as needed

# Store the features in a dictionary to be used in machine learning models
features = {'average_line_length': avg_length, 'average_line_angle': avg_angle}


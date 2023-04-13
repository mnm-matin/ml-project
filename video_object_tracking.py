python
# Import necessary libraries
import cv2
import numpy as np


# Load video file
cap = cv2.VideoCapture('video.mp4')


# Initialize variables
prev_frame = None
object_positions = []


# Define function to track object movement
def track_object(frame):
    global prev_frame, object_positions

    # Convert frame to grayscale for processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If this is the first frame, set it as the previous frame and return
    if prev_frame is None:
        prev_frame = gray_frame
        return

    # Calculate the difference between the current frame and previous frame
    frame_diff = cv2.absdiff(gray_frame, prev_frame)

    # Apply thresholding to the difference image
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Apply morphological operations (dilation followed by erosion) to remove noise
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return
    if len(contours) == 0:
        return

    # Otherwise, loop over the contours and find the largest one
    max_contour = max(contours, key=cv2.contourArea)

    # Find the centroid of the largest contour
    M = cv2.moments(max_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Add the centroid to the list of object positions
    object_positions.append((cx, cy))

    # Update the previous frame with the current frame
    prev_frame = gray_frame


# Loop over each frame in the video
while True:
    # Read in the next frame
    ret, frame = cap.read()

    # If there are no more frames, break out of the loop
    if not ret:
        break

    # Track the movement of any objects in the frame
    track_object(frame)

    # Display the current frame
    cv2.imshow('Frame', frame)

    # If the 'q' key is pressed, break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()


# Print the list of object positions over time
print(object_positions)

# This is an example code for controlling applications through hand gestures and movements using machine learning and OpenCV library.

# Importing necessary libraries
import cv2
import numpy as np
import pyautogui

# Initializing the video stream
cap = cv2.VideoCapture(0)
# Creating a BackgroundSubtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Creating a dictionary to map the gestures to actions
gesture_actions = {
    "fist": "left_click", 
    "palm": "right_click"
}

# Defining the hand region of interest (ROI)
hand_roi = (100, 100, 300, 300)

# Load the trained machine learning model
model = ...

while(True):
    # Reading the video stream frame by frame
    ret, frame = cap.read()
    
    # Converting the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Applying the Background Subtraction technique
    fgmask = fgbg.apply(gray)
    
    # Applying morphological transformations to remove noise
    kernel = np.ones((5,5),np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # Creating the hand region of interest (ROI)
    x,y,w,h = hand_roi
    hand_roi_frame = fgmask[y:y+h, x:x+w]
    
    # Rescaling the region of interest (ROI) to the size expected by the model
    hand_roi_rescaled = cv2.resize(hand_roi_frame, (224, 224))
    
    # Making a prediction using the trained model
    prediction = model.predict(hand_roi_rescaled)
    
    # Selecting the gesture with the highest probability
    predicted_gesture = np.argmax(prediction)
    
    # Mapping the gesture to an action
    gesture_action = gesture_actions[predicted_gesture]
    
    # Executing the action
    if gesture_action == "left_click":
        pyautogui.click(button='left')
    elif gesture_action == "right_click":
        pyautogui.click(button='right')
    
    # Displaying the frame with the hand ROI
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow('frame',frame)
    
    # Exiting the program on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the video stream and closing all windows
cap.release()
cv2.destroyAllWindows()
# Libraries Required
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Setting up mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Setting up Screen resolution
screen_res = (1920, 1080)

# Initializing PyAutoGUI 
pyautogui.FAILSAFE = False

# Initializing Holistic Model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Initializing Video Capture Module
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        
        # Converting Image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calling holistic model to get detections
        results = holistic.process(image)
        
        # Extracting Right Hand Landmarks
        right_landmarks = results.right_hand_landmarks
        
        # Setting up PyAutoGUI Screen Resolution
        pyautogui.size(screen_res)
        
        # PyAutoGUI Movement Functions
        def move_right():
            pyautogui.moveRel(75, 0, duration=0.2)
        def move_left():
            pyautogui.moveRel(-75, 0, duration=0.2)
        def move_up():
            pyautogui.moveRel(0, -75, duration=0.2)
        def move_down():
            pyautogui.moveRel(0, 75, duration=0.2)
        
        # Checking for Hand Landmarks
        if right_landmarks:
            
            # Extracting Fingertip Coordinates
            fingertip_x = right_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * screen_res[0]
            fingertip_y = right_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * screen_res[1]
            
            # Checking if Index Fingertip Raised or lowered
            if right_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y > right_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y:
                    
                # Checking Right or Left Swipe
                if right_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x < right_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x:
                    move_right()
                elif right_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x > right_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x:
                    move_left()
                    
                # Checking Up or Down Swipe
                elif right_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y < right_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y:
                    move_up()
                elif right_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y > right_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y:
                    move_down()
        
        # Drawing the Hand Landmarks and Connection Lines
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Converting Image back to BGR to show on screen
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Displaying the result
        cv2.imshow('Gesture Control', image)
        
        # Exit Condition
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    # Release Resources
    cap.release()
    cv2.destroyAllWindows()

# Code Explanation:

# We begin by importing the required libraries: cv2, mediapipe, numpy, and pyautogui.

# We then set up the mediapipe model for holistic detection.

# We initialize the PyAutoGUI module and set the screen resolution to that of the computer screen.

# We extract landmarks of the right hand from the detections obtained from the mediapipe model.

# We define movement functions for PyAutoGUI which would help us in moving the cursor around the screen.

# We then check for the presence of hand landmarks, finger and thumb tips in the extracted landmarks.

# Based on the position of the thumb and index finger, we determine whether the hand gesture represents a horizontal or vertical gesture.

# We then use PyAutoGUI to move the cursor in the respective direction based on the detected hand gesture.

# Finally, we display the video stream with detected landmarks and gesture directions.

# The video stream can be exited by pressing the 'q' key.
# Import the required Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

# Load the pre-trained model
model = load_model('vehicle_detection_model.h5')

# Define a function to predict the vehicle type
def predict_vehicle_type(image_path):
  # Load and preprocess the image
  img = image.load_img(image_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_preprocessed = preprocess_input(img_array)

  # Predict the vehicle type
  prediction = model.predict(img_preprocessed)
  if prediction[0][0] == 1:
    return "Car"
  elif prediction[0][1] == 1:
    return "Motorbike"
  elif prediction[0][2] == 1:
    return "Bus"
  elif prediction[0][3] == 1:
    return "Truck"
  else:
    return "Cannot predict"

# Define a function to detect vehicles in an image or video
def detect_vehicles(input_path, output_path, is_video=False):
  if is_video:
    # Open the video file
    video_in = cv2.VideoCapture(input_path)
    
    # Get the video properties
    fps = int(video_in.get(cv2.CAP_PROP_FPS))
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create the output video file
    video_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Loop through the frames of the video
    while True:
      # Read the next frame
      ret, frame = video_in.read()
      if not ret:
        break
      
      # Detect the vehicles in the frame
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      vehicles = vehicle_cascade.detectMultiScale(gray, 1.1, 4)
      for (x,y,w,h) in vehicles:
        # Draw a rectangle around each vehicle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Predict the type of each vehicle
        vehicle_type = predict_vehicle_type(frame[y:y+h,x:x+w])
        
        # Display the predicted vehicle type above the rectangle
        cv2.putText(frame,vehicle_type,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
        
      # Display the frame
      cv2.imshow('video',frame)
      video_out.write(frame)
      
      # Stop the video if the 'q' key is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
    # Release the video input and output files
    video_in.release()
    video_out.release()
    cv2.destroyAllWindows()
  else:
    # Load the image file
    img = cv2.imread(input_path)
    
    # Detect the vehicles in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vehicles = vehicle_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in vehicles:
      # Draw a rectangle around each vehicle
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      
      # Predict the type of each vehicle
      vehicle_type = predict_vehicle_type(img[y:y+h,x:x+w])
      
      # Display the predicted vehicle type above the rectangle
      cv2.putText(img,vehicle_type,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
    
    # Save the output image
    cv2.imwrite(output_path, img)


# Load the vehicle detection cascade classifier
vehicle_cascade = cv2.CascadeClassifier('vehicle_detection.xml')

# Call the function to detect vehicles in an image
detect_vehicles('test_image.jpg', 'output_image.jpg')

# Call the function to detect vehicles in a video
detect_vehicles('test_video.mp4', 'output_video.mp4', is_video=True)
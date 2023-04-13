
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model for colorization
model = tf.keras.models.load_model('color_model.h5')

# Read the input image
img = cv2.imread('input.png')

# Convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize the grayscale image to match the input size of the model
gray_resized = cv2.resize(gray, (256, 256))

# Normalize the grayscale image
gray_normalized = gray_resized.astype('float32') / 255.0

# Add another dimension to the grayscale image
gray_normalized = np.expand_dims(gray_normalized, axis=2)

# Predict the color for the grayscale image using the pre-trained model
colorized = model.predict(np.array([gray_normalized]))

# Remove the extra dimension from the predicted color image
colorized = np.squeeze(colorized, axis=0)

# Rescale the colorized image to the range [0, 255]
colorized_rescaled = (255 * colorized).astype(np.uint8)

# Resize the colorized image to match the size of the input image
colorized_resized = cv2.resize(colorized_rescaled, img.shape[:2][::-1])

# Display the colorized image
cv2.imshow('Colorized Image', colorized_resized)
cv2.waitKey(0)
cv2.destroyAllWindows() 

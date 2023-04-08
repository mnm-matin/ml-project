
# Import necessary libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load pre-trained image and text models
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
text_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Define a function to preprocess image and generate feature vector
def process_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    features = image_model.predict(x)
    return features

# Define a function to generate textual description of an image
def generate_description(image_path):
    features = process_image(image_path)
    description = " ".join(text_model([str(features)])[0].numpy().tolist())
    return description

# Test the function on an example image
description = generate_description("example.jpg")
print(description)

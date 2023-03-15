
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image

# Load images
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  # Load and resize image
  img = tf.io.decode_image(image_url, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  shape = tf.shape(img)[:-1]
  if preserve_aspect_ratio:
    new_shape = tf.cast(tf.math.round(tf.cast(shape, tf.float32) * tf.math.reduce_min(image_size / shape)), tf.int32)
  else:
    new_shape = image_size
  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

# Display image
def display_image(image):
  if len(image.shape) == 4:
    img = tf.squeeze(image, axis=0)
  PIL.Image.fromarray(np.array(img * 255, dtype=np.uint8)).show()

# Load the style and content images
content_image_url = tf.keras.utils.get_file('content.jpg', 'IMAGE_URL')
style_image_url = tf.keras.utils.get_file('style.jpg', 'IMAGE_URL')
content_image = load_image(content_image_url)
style_image = load_image(style_image_url)

# Display the content and style images
display_image(content_image)
display_image(style_image)

# Define the hub module
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

# Stylize the content image with the style image
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

# Display the stylized image
display_image(stylized_image)

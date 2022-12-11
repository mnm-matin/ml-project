# Required Libraries
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import PIL.Image

# Load the pre-trained StyleGAN model from Tensorflow Hub
model_url = 'https://tfhub.dev/google/progan-128/1'
module = hub.load(model_url)

# Define a function to generate an image from the given latent vectors
def generate_image(latent_vector):
    latent_vector = tf.cast(latent_vector, tf.float32)
    latent_vector = tf.reshape(latent_vector, [1, 512])
    output_image = module(latent_vector)
    output_image = tf.image.convert_image_dtype(output_image, dtype=tf.uint8, saturate=True)
    output_image = output_image[0]
    
    return PIL.Image.fromarray(output_image.numpy())

# Define a function to display the given image
def show_image(image, save_path=None):
    plt.imshow(image)
    plt.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

# Define the number of images to generate
num_images = 5

# Generate random latent vectors
latent_vectors = np.random.randn(num_images, 512)

# Generate images based on the latent vectors
generated_images = [generate_image(latent_vector) for latent_vector in latent_vectors]

# Display and save the generated images
for i in range(num_images):
    show_image(generated_images[i], save_path=f'generated_image_{i}.png')
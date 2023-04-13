# Import Libraries
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19

# Load Images
content_image = load_img('content_image.jpg')
style_image = load_img('style_image.jpg')

# Convert Images to Array
content_array = img_to_array(content_image)
style_array = img_to_array(style_image)

# Expand the Dimensions of the Image
content_array = tf.expand_dims(content_array, axis=0)
style_array = tf.expand_dims(style_array, axis=0)

# Load Pre-Trained VGG19 Model
vgg_model = vgg19.VGG19(include_top=False, weights='imagenet')

# Define Layers for Style Transfer
style_layer_names = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

content_layer_name = 'block5_conv2'

# Extract Features from Content and Style Images
content_features = vgg_model(content_array)[content_layer_name]
style_features = [vgg_model(style_array)[layer] for layer in style_layer_names]

# Compute Gram Matrix for Style Features
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# Define Style Weightage Matrix
style_weightage_matrix = [1.0 for _ in style_features]

# Define Content and Style Image Weightages
content_weightage = 0.025
style_weightage = 1.0
total_variation_weightage = 30

# Define Loss Function
def style_content_loss(outputs):
    content_outputs = outputs['content']
    style_outputs = outputs['style']
    content_loss = tf.reduce_mean(tf.square(content_outputs - content_features))
    style_loss = tf.add_n([tf.reduce_mean(tf.square(gram_matrix(style_outputs[i]) - gram_matrix(style_features[i]))) * style_weightage_matrix[i] for i in range(len(style_outputs))])
    total_variation_loss = tf.image.total_variation(outputs['combined_image']) * total_variation_weightage
    loss = content_weightage * content_loss + style_weightage * style_loss + total_variation_loss
    return loss

# Generate Synthetic Image
# Random Initialization
initial_combined_image = tf.random.uniform(tf.shape(content_array) * 255, maxval=255)
# Optimize for Style and Content Losses
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
outputs = {'combined_image': initial_combined_image}
outputs['content'] = vgg_model(outputs['combined_image'])[content_layer_name]
outputs['style'] = [vgg_model(outputs['combined_image'])[layer] for layer in style_layer_names]
for i in range(500):
    opt.minimize(lambda x: style_content_loss(outputs), var_list=[outputs['combined_image']])
    if i % 50 == 0:
        print('Iteration:', i)
        img = outputs['combined_image'].numpy()
        img = tf.squeeze(img, axis=0)
        img = tf.clip_by_value(img, 0, 255)
        img = tf.cast(img, tf.uint8)
        output_image = tf.keras.preprocessing.image.array_to_img(img)
        output_image.save(f'output_image_{i}.jpg')
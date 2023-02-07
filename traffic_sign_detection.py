# Import libraries
import tensorflow as tf
import numpy as np
import cv2

# Load the dataset
data = np.load("traffic_sign_dataset.npy")

# Split the dataset into train and test sets
X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256, activation="relu"),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=43, activation="softmax")
])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("traffic_sign_model.h5")

# Load the trained model
model = tf.keras.models.load_model("traffic_sign_model.h5")

# Load the traffic sign image
img = cv2.imread("traffic_sign.jpg")

# Resize the image
img = cv2.resize(img, (32, 32))

# Preprocess the image
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Use the model to predict the traffic sign
prediction = model.predict(img)

# Print the predicted traffic sign
print("Predicted traffic sign:", np.argmax(prediction))
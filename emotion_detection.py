# This is a sample code for a machine learning project that can detect emotions in text, speech or images

# Import required packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Embedding, Input
from tensorflow.keras.preprocessing import sequence, image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.tokenize import word_tokenize
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('emotions_dataset.csv')

# Split the data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Tokenize the text data
X['text'] = X['text'].apply(lambda x: word_tokenize(x))

# Convert the text data into sequences
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X['text'])
sequences = tokenizer.texts_to_sequences(X['text'])
X = sequence.pad_sequences(sequences, maxlen=max_len)

# Convert the target labels into categorical form
y = pd.get_dummies(y).values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
inputs = Input(shape=(max_len,))
embedding_layer = Embedding(max_words, 128, input_length=max_len)(inputs)
x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(embedding_layer)
x = Dense(64, activation='relu')(x)
outputs = Dense(6, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define earlystopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model_history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model on test data
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", confusion_mat)

# Load the image data
img = image.load_img('test_image.jpg', target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

# Use the model to predict emotions in the image
predictions = model.predict(images)
emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Fear']
predicted_emotion = emotions[np.argmax(predictions)]
print("Predicted Emotion: ", predicted_emotion)

# Load the speech data
audio_file = 'test_audio.wav'
with open(audio_file, 'rb') as f:
    audio_content = f.read()
    
# Use the model to predict emotions in the speech
# Convert the speech data into features using MFCC
features = extract_features(audio_content)
features = np.expand_dims(features, axis=0)
predictions = model.predict(features)
emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Fear']
predicted_emotion = emotions[np.argmax(predictions)]
print("Predicted Emotion: ", predicted_emotion)
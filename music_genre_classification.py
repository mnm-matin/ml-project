python
# Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical

# Reading the Metadata and Extracting Features
metadata = pd.read_csv('data.csv')
genres = metadata[['genre']]
features = []
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath('songs'), row['filename'])
    class_label = row['genre']
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    features.append([mfccs, class_label])

# Creating DataFrame for Features
df = pd.DataFrame(features, columns=['feature', 'class_label'])

# Encoding the Labels
le = LabelEncoder()
labels = le.fit_transform(df['class_label'])
classes = list(le.classes_)

# Scaling the Features
scaler = StandardScaler()
scaled_features = np.array(list(df['feature']))
scaled_features = scaler.fit_transform(scaled_features)

# Splitting the Dataset into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# One-hot Encoding the Labels
num_labels = len(classes)
y_train = to_categorical(y_train, num_labels)
y_test = to_categorical(y_test, num_labels)

# Model Building
model = Sequential()
model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Model Compilation
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Model Training
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# Plotting the Accuracy and Loss Curves
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

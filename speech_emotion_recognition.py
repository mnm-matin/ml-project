# Import the required libraries
import librosa
import numpy as np
import joblib
from sklearn.svm import SVC

# Load the dataset
X_train = np.load("train_features.npy")
y_train = np.load("train_labels.npy")

# Train the machine learning model
svm = SVC(kernel='linear', C=1, gamma='auto')
svm.fit(X_train, y_train)

# Load the test data
audio_file_path = "test.wav"
audio_data, sr = librosa.load(audio_file_path)

# Extract features from the audio data
features = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
features = np.mean(features.T, axis=0)

# Use the trained model to predict the emotional state of the speaker
predicted_label = svm.predict(features.reshape(1, -1))

# Print the predicted label
print(predicted_label)

# Save the trained model
joblib.dump(svm, "emotion_recognition_model.pkl")
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# Load audio files for both speakers
voice1, sr1 = librosa.load('voice1.wav', sr=None)
voice2, sr2 = librosa.load('voice2.wav', sr=None)

# Get the length of the shortest audio file
min_length = min(len(voice1), len(voice2))

# Trim audio files to the same length
voice1 = voice1[:min_length]
voice2 = voice2[:min_length]

# Extract features from audio files
mfcc1 = librosa.feature.mfcc(y=voice1, sr=sr1)
mfcc2 = librosa.feature.mfcc(y=voice2, sr=sr2)

# Normalize features
mfcc1 = (mfcc1 - np.mean(mfcc1)) / np.std(mfcc1)
mfcc2 = (mfcc2 - np.mean(mfcc2)) / np.std(mfcc2)

# Randomly split mfcc2 into training and testing sets
training_samples = 300
testing_samples = 100
training_indices = np.random.choice(mfcc2.shape[1], training_samples, replace=False)
testing_indices = np.setdiff1d(np.arange(mfcc2.shape[1]), training_indices)
X_train = mfcc2[:, training_indices].T
X_test = mfcc2[:, testing_indices].T

# Create labels for training and testing sets
y_train = np.zeros(training_samples)
y_test = np.zeros(testing_samples)

# Train a logistic regression model on training data
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate logistic regression model on testing data
score = lr.score(X_test, y_test)
print(f'Test accuracy: {score:.3f}')

# Predict the voice of speaker 1 using the trained model and the features of speaker 1
predicted_features = lr.predict_proba(mfcc1.T)[:, 1]

# Apply predicted features to original audio file of speaker 2 to convert it to the voice of speaker 1
predicted_audio = librosa.feature.inverse.mfcc_to_audio(predicted_features.reshape((mfcc1.shape[1], mfcc1.shape[0])), sr=sr1)

# Save the converted audio file
sf.write('voice2_converted.wav', predicted_audio, sr1)

# Display original and converted audio files
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
librosa.display.waveplot(voice1, sr=sr1)
plt.title('Voice of Speaker 1')
plt.subplot(1, 2, 2)
librosa.display.waveplot(predicted_audio, sr=sr1)
plt.title('Voice of Converted Speaker 2')
plt.tight_layout()
plt.show()
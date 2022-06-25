python
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load audio files and extract features
def extract_features(file_path):
    X, sample_rate = librosa.load(file_path)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast])

# Prepare dataset
speech_files = ["speech1.wav", "speech2.wav", "speech3.wav"]
music_files = ["music1.wav", "music2.wav", "music3.wav"]
X = []
y = []
for file_path in speech_files:
    X.append(extract_features(file_path))
    y.append("speech")
for file_path in music_files:
    X.append(extract_features(file_path))
    y.append("music")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train SVM model and evaluate accuracy
svm = SVC()
svm.fit(X_train, y_train)
accuracy = svm.score(X_test, y_test)
print("Accuracy: {:.2f}".format(accuracy))

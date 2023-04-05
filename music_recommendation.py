
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Load dataset
music_data = pd.read_csv('music_data.csv')

# Split data into training and testing sets
train_set, test_set = train_test_split(music_data, test_size=0.2, random_state=42)

# Create a Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5)

# Fit the model to the training data
model.fit(train_set)

# Define a function to recommend music tracks based on user's listening history
def recommend_music(user_history):
    # Transform user's listening history into a DataFrame
    user_history_df = pd.DataFrame(user_history, columns=['Artist', 'Track'])
    
    # Find the nearest neighbors to the user's listening history
    distances, indices = model.kneighbors(user_history_df)
    
    # Display the recommended tracks
    recommended_tracks = music_data.iloc[indices[0]]
    return recommended_tracks

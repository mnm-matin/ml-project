# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset of past user preferences
user_data = pd.read_csv('user_data.csv')

# Split dataset into train and test sets
train_data, test_data = train_test_split(user_data, test_size=0.2)

# Vectorize user preferences using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorized_train_data = vectorizer.fit_transform(train_data['preferences'])
vectorized_test_data = vectorizer.transform(test_data['preferences'])

# Calculate cosine similarity between user preferences
similarity_matrix = cosine_similarity(vectorized_test_data, vectorized_train_data)

# Function to suggest products/movies/shows to users based on their past preferences
def suggest_recommendations(user_index, num_recommendations=5):
   similarity_scores = list(enumerate(similarity_matrix[user_index]))
   similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
   top_recommendations_index = [i[0] for i in similarity_scores[0:num_recommendations]]
   return train_data['product_name'].iloc[top_recommendations_index]

# Test function by suggesting 5 recommendations to user with index 10
print(suggest_recommendations(10, num_recommendations=5))
# Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('products.csv')

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['description'], df['category'], test_size=0.2)

# Convert text descriptions into vectors using TF-IDF
tfidf = TfidfVectorizer()
X_train_vectors = tfidf.fit_transform(X_train)
X_test_vectors = tfidf.transform(X_test)

# Calculate cosine similarity between all vector pairs
cosine_sim_matrix = cosine_similarity(X_test_vectors, X_train_vectors)

# Function to recommend products to users along with explanations
def recommend_products(user_input):
  # Convert user input into a vector using the same TF-IDF model
  user_input_vector = tfidf.transform([user_input])
  # Calculate cosine similarity between user input and all product descriptions
  cosine_sim_user = cosine_similarity(user_input_vector, X_train_vectors)
  # Identify indices of the top 5 most similar products to the user input
  indices = np.argsort(cosine_sim_user[0])[:-6:-1]
  # Create list of recommended products and their corresponding category along with an explanation for the recommendation
  products = []
  for idx in indices:
    product = df.iloc[idx]['product']
    category = df.iloc[idx]['category']
    description = df.iloc[idx]['description']
    explanation = f"This recommendation is based on the similarity between your input and the following product description: {description}"
    products.append((product, category, explanation))
  return products

# Example Usage
user_input = "I am looking for a lightweight laptop for travel"
recommendations = recommend_products(user_input)
print("Recommended Products:")
for product, category, explanation in recommendations:
  print(f"\nProduct: {product}\nCategory: {category}\nExplanation: {explanation}")
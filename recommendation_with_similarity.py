# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the dataset
data = pd.read_csv('user_interaction_data.csv')

# Pre-processing the data
tfidf = TfidfVectorizer(stop_words='english')
data['text'] = data['product_name'] + ' ' + data['category']
tfidf_matrix = tfidf.fit_transform(data['text'])

# Calculating the cosine similarity between the products
similarity_matrix = cosine_similarity(tfidf_matrix)

# Function to get the top n similar products
def get_similar_products(product_id, n):
    product_index = data[data['product_id']==product_id].index[0]
    similar_products_indices = similarity_matrix[product_index].argsort()[:-n-2:-1]
    similar_products = data.iloc[similar_products_indices]
    return similar_products

# Example usage
similar_products = get_similar_products('P1001', 5)
print(similar_products[['product_name', 'category']])
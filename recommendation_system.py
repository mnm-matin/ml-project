# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Load the purchase history data
purchase_df = pd.read_csv('purchase_history.csv')

# Create a pivot table with user IDs as rows and product IDs as columns
pivot_table = pd.pivot_table(purchase_df, values='purchase_count', index='user_id', columns='product_id')

# Fill missing values with 0
pivot_table = pivot_table.fillna(0)

# Convert pivot table into sparse matrix
sparse_matrix = csr_matrix(pivot_table.values)

# Compute cosine similarities between the products
cosine_similarities = cosine_similarity(sparse_matrix)

# Create a dictionary to map product IDs to indices
products = pivot_table.columns.tolist()
product_indices = dict(zip(products, range(len(products))))

# Define a function to get similar products
def get_similar_products(product_id, num_similar_products=5):
    # Get the index of the product
    product_index = product_indices[product_id]

    # Get the cosine similarities of the product with others
    similarities = cosine_similarities[product_index]

    # Get the indices of top similar products
    similar_indices = similarities.argsort()[:-num_similar_products-1:-1]

    # Get the IDs of top similar products
    similar_products = [products[i] for i in similar_indices]

    return similar_products

# Test the function
get_similar_products('product1')
# Output: ['product2', 'product3', 'product4', 'product5', 'product6']
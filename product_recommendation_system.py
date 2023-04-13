
# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# load the dataset
df = pd.read_csv("product_ratings.csv")

# create a user-product rating matrix
user_ratings = df.pivot_table(index='user_id', columns='product_id', values='rating')

# calculate cosine similarity matrix
item_similarity_matrix = cosine_similarity(user_ratings.T)

# define a function to return recommended products
def get_recommendations(product_id):
    # get similarity scores
    similarity_scores = list(enumerate(item_similarity_matrix[product_id]))

    # sort by similarity score and select top 5
    top_similar_products = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]

    # extract product ids from the list of top similar products
    recommended_product_ids = [i[0] for i in top_similar_products]

    # return recommended products
    return recommended_product_ids

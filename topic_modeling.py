# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Import the dataset
data = pd.read_csv("dataset.csv")

# Tokenize the documents
tokenized_docs = data['text'].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokenized_docs = [[token.lower() for token in doc if token.lower() not in stop_words] for doc in tokenized_docs]

# Create a dictionary of all the words in the dataset
dictionary = [word for doc in tokenized_docs for word in doc]

# Find the most frequent words
freq_dist = FreqDist(dictionary)
most_common = freq_dist.most_common(100)

# Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, max_df=0.5)
X = vectorizer.fit_transform(data['text'])

# Use Latent Dirichlet Allocation to discover latent topics
lda = LatentDirichletAllocation(n_components=10, random_state=0)

# Train the model on the vectorized documents
lda.fit(X)

# Get the top words for each topic
words = vectorizer.get_feature_names()
for i, topic in enumerate(lda.components_):
    top_words = [words[i] for i in topic.argsort()[:-11:-1]]
    print(f"Topic {i}: {', '.join(top_words)}")
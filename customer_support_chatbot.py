# First, import necessary libraries

import nltk
import numpy as np
import random
import string

# Then, load and preprocess the dataset

dataset_file = 'dataset.txt'

with open(dataset_file, 'r') as file:
    dataset = file.read()

# Tokenize the dataset
sent_tokens = nltk.sent_tokenize(dataset)
word_tokens = nltk.word_tokenize(dataset)

# Preprocess the dataset
lemmer = nltk.stem.WordNetLemmatizer()

def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmer.lemmatize(token.lower()) for token in tokens]
    return tokens

# Create the bag of words

def create_bag_of_words(tokens):
    bag = {}
    for token in tokens:
        if token in string.punctuation:
            continue
        if token in bag:
            bag[token] += 1
        else:
            bag[token] = 1
    return bag

# Create the bag of words for the dataset
word_bag = create_bag_of_words(word_tokens)

# Define function to return response to user input

def response(user_input):

    # Create the bag of words for the user input
    user_input_tokens = preprocess(user_input)
    user_input_bag = create_bag_of_words(user_input_tokens)

    # Compute the similarity score between the user input and each sentence in the dataset using the cosine similarity algorithm
    scores = {}
    for sent in sent_tokens:
        sent_tokens = preprocess(sent)
        sent_bag = create_bag_of_words(sent_tokens)
        score = 0
        for token in user_input_bag:
            if token in sent_bag:
                score += sent_bag[token] * user_input_bag[token]
        scores[sent] = score

    # Return the sentence with the highest similarity score, as a response to the user input
    return max(scores, key=scores.get)

# Define function to chat with the user

def chat():
    print('Hello! How can I assist you today?')
    while True:
        user_input = input().lower()
        if user_input == 'quit':
            break
        else:
            print(response(user_input))

# Run the chatbot
chat()
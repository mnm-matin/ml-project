# This is a Python code to build a chatbot that matches the brand's personality

# Import necessary libraries
import nltk
import numpy as np
import random
import string
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a function to choose a response from the bot
def bot_response(user_input):
    sent_tokens.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        bot_response = "I am sorry! I don't understand you"
        return bot_response
    else:
        bot_response = sent_tokens[idx]
        return bot_response

# Define a function to tokenize and normalize the text
def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

# Define the main function
def chatbot():
    # Feel free to add more sentences to this list
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad we are talking"]
    BRAND_RESPONSES = ["Our brand stands for quality, innovation, and customer satisfaction",
                       "We believe in delivering the best products and services to our customers",
                       "We are committed to building lasting relationships with our customers",
                       "Our brand is all about making a positive impact on the world",
                       "We are dedicated to creating a culture of excellence and integrity",
                       "Our mission is to provide the best possible experience for our customers"]
    # Greet the user
    print("Welcome to the chatbot! How may I assist you?")
    while(True):
        user_input = input()
        if(user_input.lower() in GREETING_INPUTS):
            bot_response = random.choice(GREETING_RESPONSES)
        elif('brand' in user_input.lower()):
            bot_response = random.choice(BRAND_RESPONSES)
        else:
            bot_response = bot_response(user_input)
        print(bot_response)

# Run the main function
chatbot()
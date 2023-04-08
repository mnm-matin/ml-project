# Here's a code example that uses the Python Natural Language Toolkit (NLTK) library
# and Markov chains to create a text generator that can generate new text based on existing text.

import nltk
import random

# Read in some sample text
with open('sample_text.txt', 'r') as file:
    sample_text = file.read()

# Pre-process the text with NLTK
tokens = nltk.word_tokenize(sample_text)
processed_text = [word.lower() for word in tokens if word.isalpha()]

# Generate a frequency distribution of words
freq_dist = nltk.FreqDist(processed_text)

# Generate a transition matrix for Markov chains
transition_matrix = {}
for i in range(len(processed_text)-1):
    cur_word = processed_text[i]
    next_word = processed_text[i+1]
    if cur_word not in transition_matrix:
        transition_matrix[cur_word] = {}
    if next_word not in transition_matrix[cur_word]:
        transition_matrix[cur_word][next_word] = 0
    transition_matrix[cur_word][next_word] += 1

# Normalize the transition matrix
for cur_word in transition_matrix:
    total_transitions = sum(transition_matrix[cur_word].values())
    for next_word in transition_matrix[cur_word]:
        transition_matrix[cur_word][next_word] /= total_transitions

# Use the transition matrix to create a new sentence
new_sentence = []
cur_word = random.choice(tuple(transition_matrix.keys()))

while True:
    new_sentence.append(cur_word)
    if len(new_sentence) > 10 and cur_word.endswith('.'):
        break
    probabilities = transition_matrix[cur_word]
    next_word = random.choices(list(probabilities.keys()), 
                               list(probabilities.values()))[0]
    cur_word = next_word

generated_text = ' '.join(new_sentence)
print(generated_text)
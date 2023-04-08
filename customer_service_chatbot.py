# Import necessary libraries
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Define training data
training_data = [
{"class":"greeting", "sentence":"Hi, how can I help you?"},
{"class":"greeting", "sentence":"Hello, what can I do for you?"},
{"class":"greeting", "sentence":"Hi there, how can I assist you?"},
{"class":"goodbye", "sentence":"Goodbye! Have a nice day!"},
{"class":"goodbye", "sentence":"Thank you and come back soon."},
{"class":"goodbye", "sentence":"Bye, see you soon!"},
{"class":"problem", "sentence":"I have an issue with my order."},
{"class":"problem", "sentence":"My order hasn't arrived yet."},
{"class":"problem", "sentence":"The product I received is defective."},
{"class":"product_info", "sentence":"What are the features of this product?"},
{"class":"product_info", "sentence":"What is the price of this product?"},
{"class":"product_info", "sentence":"Is this product available in different colors?"}
]

# Extracting training data
corpus_words = {}
class_words = {}
classes = list(set([a['class'] for a in training_data]))
for c in classes:
    class_words[c] = []

for data in training_data:
    # Tokenizing each sentence into words
    for word in nltk.word_tokenize(data['sentence']):
        # Stemming each word
        word = stemmer.stem(word.lower())
        if word not in ["?", "'s"]:
            if word not in corpus_words:
                corpus_words[word] = 1
            else:
                corpus_words[word] += 1
            class_words[data['class']].extend([word])

# Creating input matrix
training_inputs = []
training_outputs = []
output_empty = [0] * len(classes)
for data in training_data:
    training_sent = []
    for word in nltk.word_tokenize(data['sentence']):
        training_sent.append(stemmer.stem(word.lower()))
    output_row = list(output_empty)
    output_row[classes.index(data['class'])] = 1
    training_inputs.append(training_sent)
    training_outputs.append(output_row)

# Creating bag of words
corpus_words = sorted(list(corpus_words.keys()))
classes = sorted(list(set(classes)))
print(len(training_inputs), "sentences")
print(len(classes), "classes", classes)
print(len(corpus_words), "unique stemmed words", corpus_words)

# Creating training data
training = []
for index, sent in enumerate(training_inputs):
    bag = []
 
    for word in corpus_words:
        bag.append(1) if word in sent else bag.append(0)

    training.append([bag, training_outputs[index]])

training = np.array(training)

# Defining model
input_shape = (len(training[0][0]),)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=input_shape, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(training[0][1]), activation='softmax')
])

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting model
model.fit(training[:,0].tolist(), training[:,1].tolist(), epochs=1000, batch_size=8, verbose=1)

# Saving model
model.save('chatbot_model.h5')

# Saving classes and corpus_words
import pickle
pickle.dump({'classes':classes, 'corpus_words':corpus_words, 'stemmer_words':stemmer}, open('chatbot_classes.pkl', 'wb'))
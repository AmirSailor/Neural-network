import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
# print("Current working directory:", os.getcwd())

import tensorflow as tf
import pickle
import json
import random

from config import ignored_words

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
# nltk.download('punkt')  # Uncomment if you need to download the punkt tokenizer
# nltk.download('wordnet')  # Uncomment if you need to download the WordNet lemmatizer
# nltk.download('omw-1.4')  # Uncomment if you need to download the Open Multilingual WordNet
# Initialize lists to hold words, classes, and documents 

words = []
classes = []
documents = []


# get a list of words and add them to the total words list
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # add to documents in the corpus
        documents.append((word_list, intent['tag']))
        # add to classes if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignored_words]  # lemmatize each word and lower case it
words = sorted(set(words))  # remove duplicates and sort

classes = sorted(set(classes))  # sort classes

# save the words and classes to pickle files
pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    # lemmatize each word and lower case it
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words if word not in ignored_words]
    
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)  # shuffle the training data
# debug
# print(f"bag length: {len(bag)}, output_row length: {len(output_row)}")

# this code didnt work as expected whitout dtype=object

training = np.array(training, dtype=object)  
train_x = list(training[:, 0])  # training data
train_y = list(training[:, 1])  # training labels

# create the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(train_y[0]), activation='softmax'))
# compile the model
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True), metrics=['accuracy'])

# fit the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# save the model
model.save('model/chatbot_model.keras', save_format='keras')
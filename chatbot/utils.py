import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import pickle
import json
import random

from config import ignored_words

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

def clean_up_sentence(sentence):
    """
    Tokenizes and lemmatizes the input sentence.
    """
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() and word not in ignored_words]  # Keep only alphanumeric words and ignore ignored words
    
    return words

def bag_of_words(sentence):
    words = pickle.load(open('model/words.pkl', 'rb'))
    sentence_words = clean_up_sentence(sentence)  # Tokenize and lemmatize the input sentence
    bag = [0] * len(words)  # Initialize the bag of words
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1  # Set the index to 1 if the word is found in the sentence
    return np.array(bag)

def predict_class(sentence):
    classes = pickle.load(open('model/classes.pkl', 'rb'))
    model = load_model('model/chatbot_model.keras')
    p = bag_of_words(sentence)  # Get the bag of words representation
    res = model.predict(np.array([p]))[0]  # Predict the class probabilities
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filter results based on the error threshold
    results.sort(key=lambda x: x[1], reverse=True)  # Sort results by probability
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Create a list of dictionaries with intent and probability
    return return_list

def get_response(intents_list):
    """
    Returns a response based on the predicted intent.
    """
    intents = json.loads(open('intents.json').read())
    tag = intents_list[0]['intent']  # Get the intent from the list
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])  # Return a random response from the matched intent
    return "I didn't understand that. Can you please rephrase?"
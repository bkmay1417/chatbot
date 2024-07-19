import nltk
import json
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

from unidecode import unidecode

# Inicialización del lematizador
lemmatizer = WordNetLemmatizer()

def correct_spelling(text):
    """Corrige errores ortográficos en el texto."""
    texto =unidecode(text)
    #texto =str(TextBlob(texto).correct())
    return texto

# Cargar datos
intents = json.loads(open('intents_spanish.json', 'r', encoding='utf-8').read())
words = pickle.load(open('words_spanish.pkl', 'rb'))
classes = pickle.load(open('classes_spanish.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    """Tokeniza y lematiza la oración."""
    sentence = correct_spelling(sentence) 
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convierte la oración en una bolsa de palabras binaria."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predice la clase de la oración dada."""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Obtiene una respuesta basada en la lista de intenciones."""
    if not intents_list:
        return "Lo siento, no entiendo lo que dices."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "Lo siento, no tengo una respuesta para eso."

    return result

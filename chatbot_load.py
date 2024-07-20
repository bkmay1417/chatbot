import nltk
import json
import pickle
import numpy as np
import random
import requests
import io
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from unidecode import unidecode
import tempfile
import tensorflow as tf

# Inicialización del lematizador
lemmatizer = WordNetLemmatizer()

def correct_spelling(text):
    """Corrige errores ortográficos en el texto."""
    texto =unidecode(text)
    #texto =str(TextBlob(texto).correct())
    return texto

def load_pickle_from_url(url):
    """Carga un archivo pickle desde una URL."""
    response = requests.get(url)
    response.raise_for_status()  # Verifica que la solicitud haya sido exitosa
    return pickle.load(io.BytesIO(response.content))

def load_json_from_url(url):
    """Carga un archivo JSON desde una URL."""
    response = requests.get(url)
    response.raise_for_status()  # Verifica que la solicitud haya sido exitosa
    return response.json()

# URLs de los archivos
intents_url = 'https://github.com/bkmay1417/chatbot/raw/501f470b57007c3c8e2faa732a9848a3f5bb05f8/intents_spanish.json'  # Reemplaza <commit> con el commit correcto
words_url = 'https://github.com/bkmay1417/chatbot/raw/501f470b57007c3c8e2faa732a9848a3f5bb05f8/words_spanish.pkl'       # Reemplaza <commit> con el commit correcto
classes_url = 'https://github.com/bkmay1417/chatbot/raw/501f470b57007c3c8e2faa732a9848a3f5bb05f8/classes_spanish.pkl'   # Reemplaza <commit> con el commit correcto

# URL del modelo en GitHub
model_url = 'https://github.com/bkmay1417/chatbot/blob/6b8460de8a4b5bd8b993ce206b6efef916a49ace/chatbot_model.h5?raw=True'

# Descargar y cargar el modelo
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        return tf.keras.models.load_model(temp_file_path)
    else:
        st.error(f"Error al descargar el archivo: {response.status_code}")
        return None

model = load_model_from_url(model_url)


# Cargar archivos
intents = load_json_from_url(intents_url)
words = load_pickle_from_url(words_url)
classes = load_pickle_from_url(classes_url)


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

import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode

# Cargar el archivo JSON
data_file = open('intents_spanish.json', 'r', encoding='utf-8').read()
intents = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', '¿', '¡']

# Recorre cada intención y sus patrones en el archivo JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Quita acentos y tokeniza las palabras en cada patrón
        pattern = unidecode(pattern)
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Agrega el par (patrón, etiqueta) a la lista de documentos
        documents.append((w, intent['tag']))
        # Si la etiqueta no está en la lista de clases, la agrega
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematiza las palabras y las convierte en minúsculas, excluyendo las palabras ignoradas
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Guarda las listas de palabras y clases en archivos pickle
pickle.dump(words, open('words_spanish.pkl', 'wb'))
pickle.dump(classes, open('classes_spanish.pkl', 'wb'))

print("Archivos 'words_spanish.pkl' y 'classes_spanish.pkl' creados exitosamente.")

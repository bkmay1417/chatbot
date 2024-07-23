import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import random
from unidecode import unidecode
import re

# Función para preprocesar texto
def preprocess_text(text):
    text = text.lower()
    text = unidecode(text)  # Eliminar acentos
    text = re.sub(r'\W', ' ', text)
    return word_tokenize(text)

# Función para cargar el modelo Word2Vec
def load_word2vec_model(model_path):
    model = Word2Vec.load(model_path)
    return model

# Función para crear el embedding matrix
def create_embedding_matrix(word2vec_model, words, embedding_dim=300):
    embedding_matrix = np.zeros((len(words), embedding_dim))
    for i, word in enumerate(words):
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
        else:
            embedding_matrix[i] = np.random.normal(size=(embedding_dim,))
    return embedding_matrix

# Función para crear el modelo de chatbot
def create_chatbot_model(train_x, train_y, embedding_matrix, words, classes):
    model = Sequential()
    model.add(Embedding(input_dim=len(words), output_dim=300, weights=[embedding_matrix], input_length=len(words), trainable=False))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5')
    print("Modelo del chatbot creado y guardado como 'chatbot_model.h5'")
    return model

if __name__ == "__main__":
    with open('intents_spanish.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
    
    lemmatizer = WordNetLemmatizer()
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.', ',', '¿', '¡']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern = unidecode(pattern)
            w = preprocess_text(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    sentences = [doc[0] for doc in documents]
    word2vec_model = load_word2vec_model('word2vec_chatbot.model')
    embedding_matrix = create_embedding_matrix(word2vec_model, words)

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = np.zeros(len(words))
        pattern_words = doc[0]
        for word in pattern_words:
            if word in words:
                bag[words.index(word)] = 1
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    train_x = np.array([row[0] for row in training])
    train_y = np.array([row[1] for row in training])

    create_chatbot_model(train_x, train_y, embedding_matrix, words, classes)

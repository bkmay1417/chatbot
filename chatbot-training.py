import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers.schedules import ExpontialDecay
import random
from unidecode import unidecode

data_file = open('intents_spanish.json','r',encoding='utf-8').read()
intents =json.loads(data_file)

lemmatizer = WordNetLemmatizer()

words=[]
classses =[]
documents= []
ignore_words = ['?','!','.',',','¿','¡','¿','¡','¿']

#recorre cada intencion y sus patrones en el archivo json
for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = unidecode(pattern)
        #tokeniza las palabras en cada patron y agrega a la lista de palabras
        w= nltk.word_tokenize(pattern)
        words.extend(w)
        #agrega el par (patron,etiqueta) a la lista de documentos
        documents.append((w,intent['tag']))
        #si la etiqueta no esta en la lista de clases la agrega
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#lematiza las palabras y las covierte en minusculas,escluyecndo las palabras ignoradas
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

#guarda las lista ded palabras y clases en archivos pickle
pickle.dump(words,open('words_spanish.pkl','wb'))
pickle.dump(classes,open('classes_spanish.pkl','wb'))

training = []
output_empty = [0] * len(classes)

#crea el conjunto de entrenamiendo
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower())for word in pattern_words]
    for word in words:
        #crea un bolsa de palabras binaria para cada patron
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row = list (output_empty)
    #crea un vector de salida con un 1 en la posicion correspondiente ala etiqueta
    output_row[classes.index(doc[1])] = 1
    training.append([bag,output_row])

#mescla aleeotaariamente el conjunto de entrenamiento
random.shuffle(training)

#divide el conjunto de entrenamienento en caresteristicas (train_x) y etiquetas (train_y)
train_x = [row[0] for row in training]
train_y = [row[1] for row in training]

train_x = np.array(train_x)
train_y = np.array(train_y)

# crea el mpdelo de red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#configura el optimizador con una tasse de aprendizaje exponencialemnte decreciente
lr_schedule = ExpontialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9,
)

sgd = SGD(learning_rate = lr_schedule,momentum=0.9, nestrov =True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

#entrena el modelo con el conjunto de entrenamiento
hist = model.fit(np.array(train_x),np.array(train_y), epochs=200, batch_size=5, verbose=1)

#guarda el modelo entrenado en un archivo h5
model.save('chatbot_model.h5', hist)
print("modedelo creado")
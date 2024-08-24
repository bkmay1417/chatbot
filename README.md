# chatbot

<div style="display: flex; align-items: center;">
<img src="https://img.shields.io/badge/STATUS-EN%20DESAROLLO-green" />
</div>
Este proyecto implementa un chatbot en español utilizando una red neuronal y diversas tecnologías de procesamiento de lenguaje natural (NLP). El proyecto incluye scripts para entrenar el modelo, cargarlo y utilizarlo en una aplicación Streamlit para ofrecer una interacción amigable con el usuario.

## Estructura del Proyecto

El proyecto contiene los siguientes archivos principales:

1. `train_word2vec.py`: Script para entrenar un modelo Word2Vec con un corpus específico.
2. `chatbot-training.py`: Script para entrenar el modelo del chatbot.
3. `chatbot_load.py`: Script para cargar y usar el modelo entrenado.
4. `chatbot-stramlit.py`: Script para integrar el chatbot en una aplicación Streamlit.
5. `requirements.txt`: Archivo que lista las dependencias necesarias para ejecutar los scripts.

## Requisitos

Para ejecutar el proyecto, necesitas tener instalado Python 3.6 o superior. Las dependencias del proyecto están listadas en el archivo `requirements.txt`.

### Instalación de Dependencias

Puedes instalar las dependencias usando pip:

```bash
pip install -r requirements.txt
```

## Uso
1. Entrenamiento del Modelo Word2Vec
El script train_word2vec.py se encarga de entrenar un modelo Word2Vec utilizando un corpus específico para mejorar el entendimiento del lenguaje natural en español.

### Ejecución
Para ejecutar el script de entrenamiento del modelo Word2Vec, utiliza el siguiente comando en tu terminal:

```
python train_word2vec.py
```
2. Entrenamiento del Modelo del Chatbot
El script chatbot-training.py entrena un modelo de red neuronal para el chatbot utilizando los datos de intents_spanish.json.

### Ejecución
Para ejecutar el script de entrenamiento del modelo del chatbot, utiliza el siguiente comando en tu terminal:

```
python chatbot-training.py
```
3. Integración del Chatbot con Streamlit
El script chatbot_streamlit.py integra el chatbot en una aplicación Streamlit, permitiendo una interacción más amigable con el usuario a través de una interfaz web.

Ejecución
Para ejecutar el script de Streamlit, utiliza el siguiente comando en tu terminal:

```
streamlit run chatbot_streamlit.py
```

## Tecnologías Utilizadas
Este proyecto utiliza una variedad de tecnologías y bibliotecas para crear un chatbot eficiente y funcional:

- **Python:** Lenguaje de programación principal utilizado para desarrollar el proyecto.
- **Streamlit:** Biblioteca para la creación de aplicaciones web interactivas.
- **TensorFlow y Keras:** Utilizadas para construir y entrenar el modelo de red neuronal.
- **NLTK (Natural Language Toolkit):** Utilizado para el procesamiento de texto, incluyendo tokenización y lematización.
- **Word2Vec:** Modelo de palabras para la creación de representaciones vectoriales de texto.
- **Gensim:** Biblioteca utilizada para implementar el modelo Word2Vec.
- **Unidecode:** Utilizada para eliminar acentos y caracteres especiales del texto.
- **requests:** Biblioteca para manejar solicitudes HTTP y cargar datos desde URLs.

## Acceso al Streamlit

Puedes acceder a la API en el siguiente enlace: [Mi Streamlit](https://chatbot-f4agujo33egcfeutr3chbe.streamlit.app)

## Desarrolladores

| [<img src="https://avatars.githubusercontent.com/u/163685041?v=4" width=115><br><sub>Michael Martinez</sub>](https://github.com/bkmay1417) |
| :---: |

Copyright (c) 2024 [Michael Martinez] yam8991@gmail.com

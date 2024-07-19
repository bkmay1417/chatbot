# chatbot

Este proyecto implementa un chatbot en español utilizando una red neuronal. El proyecto incluye scripts para entrenar el modelo, cargarlo y utilizarlo en una aplicación Streamlit.

## Estructura del Proyecto

El proyecto contiene los siguientes archivos principales:

1. `chatbot-training.py`: Script para entrenar el modelo del chatbot.
2. `chatbot_load.py`: Script para cargar y usar el modelo entrenado.
3. `chatbot-stramlit.py`: Script para integrar el chatbot en una aplicación Streamlit.
4. `requirements.txt`: Archivo que lista las dependencias necesarias para ejecutar los scripts.

## Requisitos

Para ejecutar el proyecto, necesitas tener instalado Python 3.6 o superior. Las dependencias del proyecto están listadas en el archivo `requirements.txt`.

### Instalación de Dependencias

Puedes instalar las dependencias usando pip:

```bash
pip install -r requirements.txt
```

## Uso


1. chatbot-training.py
Este script entrena un modelo de red neuronal para el chatbot utilizando los datos de intents_spanish.json.

### Ejecución
Para ejecutar el script de entrenamiento, simplemente ejecuta el siguiente comando en tu terminal:

```
python chatbot-training.py
```

2. chatbot-stramlit.py
Este script integra el chatbot en una aplicación Streamlit, permitiendo una interacción más amigable con el usuario a través de una interfaz web.

### Ejecución
Para ejecutar el script de Streamlit, simplemente ejecuta el siguiente comando en tu terminal:

```
streamlit run chatbot-stramlit.py
```

## Desarrolladores

| [<img src="https://avatars.githubusercontent.com/u/163685041?v=4" width=115><br><sub>Michael Martinez</sub>](https://github.com/bkmay1417) |
| :---: |

Copyright (c) 2024 [Michael Martinez] yam8991@gmail.com
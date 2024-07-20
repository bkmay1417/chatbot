import streamlit as st
from chatbot_load import predict_class, get_response, intents
import nltk

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('wordnet')

st.title("Asistente virtual")

# Inicializar las variables de estado
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Mostrar los mensajes guardados en el estado
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Mostrar el primer mensaje del asistente
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿cómo puedo ayudarte?")
    st.session_state.messages.append({"role": "assistant", "content": "Hola, ¿cómo puedo ayudarte?"})
    st.session_state.first_message = False

# Procesar el mensaje del usuario
if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
#implementacion ddel algoritmo de ia
    insts = predict_class(prompt)
    res = get_response(insts,intents)

    # Respuesta del asistente (ejemplo simple)
    with st.chat_message("assistant"):
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})

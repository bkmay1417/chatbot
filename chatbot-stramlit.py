import streamlit as st
from chatbot_load import predict_class, get_response, intents
import nltk

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('wordnet')

st.title("EcoSmart Solutions - Asistente Virtual")

# Inicializar las variables de estado
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True
if "user_interacted" not in st.session_state:
    st.session_state.user_interacted = False
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

# Mostrar los mensajes guardados en el estado
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Mostrar el primer mensaje del asistente
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, soy tu asistente virtual de EcoSmart Solutions. ¿En qué puedo ayudarte hoy?")
    st.session_state.messages.append({"role": "assistant", "content": "Hola, soy tu asistente virtual de EcoSmart Solutions. ¿En qué puedo ayudarte hoy?"})
    st.session_state.first_message = False

# Mostrar sugerencias solo en el primer mensaje y si el usuario no ha interactuado
if st.session_state.show_suggestions:
    st.write("**Sugerencias:**")
    suggestions = [
        "¿Cuánto cuesta instalar paneles solares en mi hogar?",
        "¿Ofrecen financiamiento para la compra de sus productos?",
        "¿Qué tipo de mantenimiento requieren los sistemas de energía renovable?",
        "¿Cuánto tiempo tarda la instalación de los paneles solares?",
        "¿Puedo combinar diferentes productos?"
    ]
    for suggestion in suggestions:
        if st.button(suggestion, key=suggestion):
            prompt = suggestion
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Implementación del algoritmo de IA
            insts = predict_class(prompt)
            res = get_response(insts, intents)
            
            # Respuesta del asistente
            with st.chat_message("assistant"):
                st.markdown(res)
            st.session_state.messages.append({"role": "assistant", "content": res})
            st.session_state.user_interacted = True
            st.session_state.show_suggestions = False  # Ocultar sugerencias después de seleccionar una
            break  # Salir del bucle después de seleccionar una sugerencia

# Procesar el mensaje del usuario
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.user_interacted = True
    st.session_state.show_suggestions = False  # Ocultar sugerencias después de la interacción del usuario
    
    # Implementación del algoritmo de IA
    insts = predict_class(prompt)
    res = get_response(insts, intents)

    # Respuesta del asistente
    with st.chat_message("assistant"):
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})

# Mostrar el horario de atención y métodos de contacto solo si el usuario no ha interactuado
if not st.session_state.user_interacted:
    with st.expander("Horario de Atención y Contacto"):
        st.markdown("""
        **Horario de Atención:**
        - Lunes a Viernes: 9:00 AM - 6:00 PM
        - Sábado: 10:00 AM - 2:00 PM
        - Domingo: Cerrado

        **Métodos de Contacto:**
        - Teléfono: +1 (800) 123-4567
        - Correo Electrónico: soporte@ecosmartsolutions.com
        - Redes Sociales: [Facebook](https://facebook.com), [Instagram](https://instagram.com), [Twitter](https://twitter.com)
        """)

# Mostrar enlaces útiles solo si el usuario no ha interactuado
if not st.session_state.user_interacted:
    with st.expander("Enlaces Útiles"):
        st.markdown("""
        - [Visita nuestro sitio web](https://ecosmartsolutions.com)
        - [Catálogo de Productos](https://ecosmartsolutions.com/catalogo)
        - [Promociones Actuales](https://ecosmartsolutions.com/promociones)
        """)

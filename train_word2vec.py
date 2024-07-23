import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import nlpaug.augmenter.word as naw
from unidecode import unidecode

nltk.download('punkt')
nltk.download('stopwords')

# Función para preprocesar texto
def preprocess_text(text):
    text = text.lower()
    text = unidecode(text)  # Eliminar acentos
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('spanish')]
    return tokens

# Función para aumentar datos utilizando sinónimos
def augment_texts(original_texts):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_texts = [aug.augment(sentence) for sentence in original_texts]
    return augmented_texts

# Función para entrenar el modelo Word2Vec
def train_word2vec(corpus_path, model_path):
    with open(corpus_path, 'r', encoding='utf-8') as file:
        corpus = file.read()
    
    tokens = preprocess_text(corpus)
    augmented_tokens = augment_texts([tokens])
    
    model = Word2Vec(sentences=augmented_tokens, vector_size=300, window=5, min_count=1, workers=4)
    model.save(model_path)
    print(f"Modelo Word2Vec guardado en {model_path}")

if __name__ == "__main__":
    corpus_path = 'intents_spanish.json'
    model_path = 'word2vec_chatbot.model'
    train_word2vec(corpus_path, model_path)

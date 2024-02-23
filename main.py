from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import re
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import nltk
from nltk.corpus import stopwords
from pydantic import BaseModel
from unidecode import unidecode
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Définir le chemin du modèle

model_path = 'glove_model_lstm'

# Charger le modèle sauvegardé
#loaded_model = tf.keras.models.load_model(model_path)


ml_models = {}  # Définir ml_models comme un dictionnaire vide
model = None  # Initialiser model à None

# Fonction pour charger le modèle de manière asynchrone
async def load_ml_model():
    global model  # Utiliser la variable model définie en dehors de la fonction
    try:
        model = load_model(model_path)
    except OSError:
        raise HTTPException(status_code=500, detail="Impossible de charger le modèle")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model asynchronously
    await load_ml_model()
    ml_models["glove_model_lstm"] = model  # Utiliser la variable model mise à jour ici
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

# Create the FastAPI app
app = FastAPI(lifespan=lifespan)

# Initialiser NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Charger le tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Padding des séquences pour avoir la même longueur
max_length = 40

def transformer_texte_en_sequence(textes, tokenizer, max_length):
    sequences = [tokenizer.texts_to_sequences([text])[0] for text in textes]
    sequences_pad = pad_sequences(sequences, maxlen=max_length)
    return sequences_pad

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input_data: InputData):
    text = input_data.text
    text_data = transformer_texte_en_sequence([text], tokenizer, max_length)
    sentiment_pred = model.predict(text_data)
    sentiment_score = sentiment_pred[0][0]
    sentiment_label = "Positive" if sentiment_score >= 0.5 else "Negative"

    return {"sentiment": sentiment_label, "score": float(sentiment_score)}
    
@app.get("/")
async def root():
    return {"message": "Hello World"}
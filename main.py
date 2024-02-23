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

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
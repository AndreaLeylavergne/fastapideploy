from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello, honey world 3.9"}
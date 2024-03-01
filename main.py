from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from keras.models import load_model


app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello, honey World 3.9"}
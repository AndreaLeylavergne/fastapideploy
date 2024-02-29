from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf


app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello, honey World 3.9"}
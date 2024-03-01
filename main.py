from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, honey World 3.9"}
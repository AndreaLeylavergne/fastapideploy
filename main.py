from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello, honey world 3.9"}
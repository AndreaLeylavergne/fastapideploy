from fastapi import FastAPI

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello, honey World 3.9"}
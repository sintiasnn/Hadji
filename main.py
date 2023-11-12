from fastapi import FastAPI

from tech.model import model_predict
from dotenv import load_dotenv

load_dotenv()

import os

app = FastAPI()

@app.get("/")
def index():
    return {"msg":"Mainpage !"}

@app.post("/predict")
def predictor(st : str):
    result = model_predict(st)
    return {"msg":result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8080)), log_level="debug")

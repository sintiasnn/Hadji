from fastapi import FastAPI

from tech.model import model_predict

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
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")

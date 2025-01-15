from api.schemas.digit_data import DigitData
from fastapi import FastAPI

from api.utils.models import load_model_and_predict

app = FastAPI(title="MNIST Digit Recognizer API",
              description="API for MNIST Digit Recognizer", version="0.1")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict_digit(data: DigitData):
    # Load the model, preprocess input, and make prediction
    return load_model_and_predict(data)
    # return {"digit": 7}  # Placeholder response

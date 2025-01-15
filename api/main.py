from api.schemas.digit_data import DigitData
from fastapi import FastAPI

app = FastAPI(title="MNIST Digit Recognizer API",
              description="API for MNIST Digit Recognizer", version="0.1")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict_digit(data: DigitData):
    # Load the model, preprocess input, and make prediction
    return {"digit": 7}  # Placeholder response

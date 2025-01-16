from http.client import HTTPException
import joblib
import numpy as np
import pandas as pd
from api.schemas.digit_data import DigitData
from pathlib import Path


def load_model(model_path):
    model = joblib.load(model_path)
    return model


def read_input_from_file():
    input_file = Path("api", "data", "user_prediction")
    input_data = pd.read_csv(input_file)
    return input_data


def load_model_and_predict(df):
    model_path = Path("api", "models", "svc_prod_v1.joblib")
    model = load_model(str(model_path))
    try:
        model_prediction = model.predict(df)
        return {"prediction": int(model_prediction[0])}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException()

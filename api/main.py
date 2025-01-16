from http.client import HTTPException
from pathlib import Path
from api.schemas.captured_input import CapturedInput
from api.schemas.digit_data import DigitData
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd

from api.utils.models import load_model_and_predict
from api.utils.preprocessing import convert_prediction_request_to_dataframe


app = FastAPI(title="MNIST Digit Recognizer API",
              description="API for MNIST Digit Recognizer", version="0.1")


app.mount("/static", StaticFiles(directory="public/static"), name="static")
templates = Jinja2Templates(directory="public")
# user_input_file = Path("api", "data", "user_input.csv")


@app.get("/", response_class=templates.TemplateResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_digit(data: DigitData):
    # convert the input data to a DataFrame
    df = convert_prediction_request_to_dataframe(data)
    # Load the model, preprocess input, and make prediction
    return load_model_and_predict(df)


# @app.post("/capture")
# def capture_digit(data: CapturedInput):
#     try:
#         # Create a DataFrame from the input
#         flattened_pixels = [pixel for row in data.pixels for pixel in row]
#         input_data = {"class": data.label, **
#                       {f"pixel{i}": v for i, v in enumerate(flattened_pixels)}}
#         df = pd.DataFrame([input_data])

#         # Save to CSV
#         if user_input_file.exists():
#             df.to_csv(user_input_file, mode="a", header=False, index=False)
#         else:
#             df.to_csv(user_input_file, mode="w", header=True, index=False)

#         return {"message": "Input captured successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

from fastapi.responses import HTMLResponse
from api.schemas.digit_data import DigitData
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from api.utils.models import load_model_and_predict

app = FastAPI(title="MNIST Digit Recognizer API",
              description="API for MNIST Digit Recognizer", version="0.1")


app.mount("/static", StaticFiles(directory="public/static"), name="static")
templates = Jinja2Templates(directory="public")


@app.get("/", response_class=templates.TemplateResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_digit(data: DigitData):
    # Load the model, preprocess input, and make prediction
    return load_model_and_predict(data)
    # return {"digit": 7}  # Placeholder response

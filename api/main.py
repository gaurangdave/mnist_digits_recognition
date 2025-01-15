from fastapi import FastAPI, HTTPException
import os
import sys

app = FastAPI(title="MNIST Digit Recognizer API",
              description="API for MNIST Digit Recognizer", version="0.1")


@app.get("/")
def read_root():
    return {"Hello": "World"}

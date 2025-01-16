# Define input schema
from pydantic import BaseModel


class CapturedInput(BaseModel):
    pixels: list[list[float]]
    label: int

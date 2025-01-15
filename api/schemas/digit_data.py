from pydantic import BaseModel


class DigitData(BaseModel):
    pixels: list[list[float]]  # 28x28 matrix for the digit image

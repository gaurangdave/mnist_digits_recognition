# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app
import pandas as pd

client = TestClient(app)


def test_hello_world_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_predict_digit_endpoint():
    payload = {"pixels": [[0.0] * 28] * 28}  # Example 28x28 input
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_digit_with_valid_input():
    mnist_test_set_path = "api/data/mnist_test_set.csv"
    # read mnist_test_set
    mnist_test_set = pd.read_csv(mnist_test_set_path)
    # read random image from the test set
    test_image = mnist_test_set.sample(1)
    # get the pixels
    pixels = test_image.drop("class", axis=1).values.tolist()
    # get the target
    target = test_image["class"].values[0]
    print("Target: ", target)
    payload = {"pixels": pixels}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

    assert response.json()["prediction"] == target

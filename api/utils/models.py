from http.client import HTTPException
import joblib
import numpy as np
from api.schemas.digit_data import DigitData


def load_model_and_predict(data: DigitData):
    model_path = "api/models/svc_prod.joblib"
    model = joblib.load(model_path)
    try:
        # Step 1: Convert the payload to a NumPy array
        # Reshape to (1, 784) for a single digit
        input_array = np.array(data.pixels).reshape(1, -1).copy()
        # Step 2: Make prediction
        probabilities = model.predict_proba(input_array)
        # Step 3: Convert probabilities to class
        prediction = np.argmax(probabilities, axis=1)[0]
        # Step 4: Return the prediction
        return {"prediction": int(prediction)}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException()

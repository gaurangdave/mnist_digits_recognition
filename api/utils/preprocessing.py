from sklearn.preprocessing import Binarizer, MinMaxScaler
import pandas as pd
from pathlib import Path
from api.schemas.digit_data import DigitData
from api.utils.common import save_image_from_dataframe


def preprocess_data(data, method="none", threshold=128):
    """
    Preprocess MNIST data based on the specified method.

    Args:
        data (pd.DataFrame): Input dataset with only features.
        method (str): Preprocessing method - "normalize", "binarize", or "none".

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    if method == "normalize":
        if len(data) == 1:
            print("Single-row input detected. Skipping normalization.")
            return data
        print("Normalizing data...")
        scaler = MinMaxScaler()
        transformed_data = scaler.fit_transform(data)
        transfored_df = pd.DataFrame(transformed_data)
        print(f"transfored_df Min: {transfored_df.min().min()}, Max: {
              transfored_df.max().max()}")
        return transfored_df
    elif method == "binarize":
        binarizer = Binarizer(threshold=threshold)
        transformed_data = binarizer.fit_transform(data)
        return pd.DataFrame(transformed_data)
    # else, keep features unchanged (no transformation)

    # Combine processed features and labels
    return pd.DataFrame(data)


def convert_prediction_request_to_dataframe(data: DigitData):
    """
    Convert the prediction request to a DataFrame.
    """
    # First flatten the 2D array to 1D
    flattened_pixels = [pixel for row in data.pixels for pixel in row]

    # Create DataFrame with pixel columns
    input_data = {f"pixel{i+1}": v for i, v in enumerate(flattened_pixels)}
    df = pd.DataFrame([input_data])

    # Debug prints
    print("DataFrame shape:", df)
    print("Sample of first 10 pixels:", df.iloc[0, :10].values)
    print("Number of non-zero pixels:", (df != 0).sum().sum())

    # Save the exact DataFrame we'll use for prediction
    prediction_request_file = Path(
        "api", "data", "user_prediction_request.csv")
    if prediction_request_file.exists():
        df.to_csv(prediction_request_file, mode="a", header=False, index=False)
    else:
        df.to_csv(prediction_request_file, mode="w", header=True, index=False)

    # # Verify the data is in the correct format for the model
    # if df.shape[1] != 784:
    #     raise ValueError(f"Expected 784 features, got {df.shape[1]}")

    return df

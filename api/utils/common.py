import matplotlib
import numpy as np
from pathlib import Path
import gdown
from joblib import dump
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # Use the Agg backend for non-GUI environments


# helper function to save the model metrics to google drive
models_path = Path("models")


def save_comparison_df(comparison_df):
    comparison_df.to_csv(
        f"{str(models_path)}/mnist_models_metrics.csv", index=False)

# helper function to dump and save the model on google drive


def save_model(estimator, file_name):
    # model path
    print(f"Saving model... {file_name}")
    model_path = Path("models", file_name)
    dump(estimator, str(model_path))


def download_from_google_drive(file_id, file_name):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, file_name, quiet=False)


def save_image_from_dataframe(df, output_file="debug_image.png"):
    """
    Save the first row of the DataFrame as an image.

    Args:
        df (pd.DataFrame): DataFrame with pixel data in columns (e.g., pixel0, pixel1, ..., pixel783).
        output_file (str): File name to save the image (default: "debug_image.png").
    """
    try:
        # Extract the first row of pixel data
        row_data = df.iloc[0].values  # Convert row to NumPy array

        # Reshape the flattened array into 28x28
        image_data = row_data.reshape(28, 28)

        # Save the image using matplotlib
        plt.imshow(image_data, cmap="gray")
        plt.axis("off")
        plt.title("Debug Image")
        plt.savefig(output_file)
        print(f"Image saved as {output_file}")
    except Exception as e:
        print(f"Error: {e}")

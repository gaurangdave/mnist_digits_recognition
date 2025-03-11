import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
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


MODELS_DIR_PATH = Path("models")
DATA_DIR_PATH = Path("data")


def download_data_from_gdrive(data_dir=DATA_DIR_PATH):
    # check if DATA_DIR_PATH exists else create it
    if not data_dir.exists():
        print(f"Creating data directory...{data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)

    # download data from gdrive folder
    gdown.download_folder(
        url="https://drive.google.com/drive/folders/10FmschultsicypMnWv-uI957F35P2ro5?usp=sharing", output=str(data_dir), quiet=False)
    print("Data downloaded successfully!")


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


def shift_image(image, direction):
    """
    Shift an MNIST image in a given direction.

    Args:
        image (np.ndarray): 1D array of 784 pixels (28x28 image).
        direction (str): One of 'up', 'down', 'left', 'right'.

    Returns:
        np.ndarray: Shifted 1D image (flattened).
    """
    # Reshape the flat image to 28x28
    image_2d = image.reshape(28, 28)
    # Create an empty 28x28 array filled with zeros
    shifted = np.zeros_like(image_2d)

    # Perform the shift based on the direction
    if direction == "up":
        shifted[:-1, :] = image_2d[1:, :]  # Shift rows up
    elif direction == "down":
        shifted[1:, :] = image_2d[:-1, :]  # Shift rows down
    elif direction == "left":
        shifted[:, :-1] = image_2d[:, 1:]  # Shift columns left
    elif direction == "right":
        shifted[:, 1:] = image_2d[:, :-1]  # Shift columns right

    # Flatten the shifted image back to 1D
    return shifted.flatten()


def augment_dataset(X, y):
    """
    Augment the MNIST dataset by creating shifted versions of each image.

    Args:
        X (pd.DataFrame): Dataset with features (flattened MNIST images).
        y (pd.Series): Labels for the images.

    Returns:
        pd.DataFrame, pd.Series: Augmented dataset (features and labels).
    """
    augmented_X = []
    augmented_y = []

    directions = ["up", "down", "left", "right"]
    total_iterations = len(X)

    # Iterate through each image and label
    with tqdm(total=total_iterations, desc="Augmenting Data") as pbar:
        for image, label in zip(X.values, y.values):
            # Append the original image
            augmented_X.append(image)
            augmented_y.append(label)

            # Create shifted images for all four directions
            for direction in directions:
                shifted_image = shift_image(image, direction)
                augmented_X.append(shifted_image)
                augmented_y.append(label)  # Same label for shifted image
            pbar.update(1)

    print("converting lists to DataFrame and Series")
    # Convert lists to DataFrame and Series
    augmented_X = pd.DataFrame(augmented_X, columns=X.columns)
    augmented_y = pd.Series(augmented_y)

    return augmented_X, augmented_y

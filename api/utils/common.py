from pathlib import Path
import gdown
from joblib import dump

# helper function to save the model metrics to google drive
models_path = Path("api", "models")


def save_comparison_df(comparison_df):
    comparison_df.to_csv(
        f"{str(models_path)}/mnist_models_metrics.csv", index=False)

# helper function to dump and save the model on google drive


def save_model(estimator, file_name):
    # model path
    print(f"Saving model... {file_name}")
    model_path = Path("api", "models", file_name)
    dump(estimator, str(model_path))

# function to download data from google drive


def download_from_google_drive(file_id, file_name):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, file_name, quiet=False)

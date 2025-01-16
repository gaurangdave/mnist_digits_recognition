from pathlib import Path
from api.utils.common import download_from_google_drive
import gdown

MODELS_DIR_PATH = Path("models")
DATA_DIR_PATH = Path("data")


def download_data_from_gdrive():
    # check if DATA_DIR_PATH exists else create it
    if not DATA_DIR_PATH.exists():
        print(f"Creating data directory...{DATA_DIR_PATH}")
        DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)

    # download data from gdrive folder
    gdown.download_folder(
        url="https://drive.google.com/drive/folders/10FmschultsicypMnWv-uI957F35P2ro5?usp=sharing", output=str(DATA_DIR_PATH), quiet=False)
    print("Data downloaded successfully!")


def download_models_from_gdrive():
    # check if MODELS_DIR_PATH exists else create it
    if not MODELS_DIR_PATH.exists():
        print(f"Creating models directory...{MODELS_DIR_PATH}")
        MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

    # download models from gdrive folder
    gdown.download_folder(
        url="https://drive.google.com/drive/folders/10GxWYi3NkoZv1Zba9yQ2y88TRM5-xKJG?usp=drive_link", output=str(MODELS_DIR_PATH), quiet=False)
    print("Models downloaded successfully!")


if __name__ == "__main__":
    print("setting up the project...")
    print("downloading data from google drive...")
    download_data_from_gdrive()
    print("downloading models from google drive...")
    download_models_from_gdrive()

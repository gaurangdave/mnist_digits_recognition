from api.utils.common import save_model, download_from_google_drive
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pathlib import Path

import pandas as pd
from datetime import datetime

# helper function to train the Support Vector Classifier (SVC) model


def train_svc(features, target, model_name):
    """ Train a Support Vector Classifier (SVC) model on the input data.
    """
    
    ## if no model name is provided, use the default name with timestamp
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"svc_prod_{timestamp}.joblib"
    
    # Best Parameters: {'svc__C': 10, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'}
    # initialize SVC
    svc = SVC(random_state=42, C=10, gamma="scale",
              kernel="rbf")

    # create pipeline
    print("Creating pipeline...")
    svc_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("svc", svc)
    ], verbose=True)

    # fit the model
    print("Fitting the model...")
    svc_pipeline.fit(features, target)

    # save the model
    save_model(svc_pipeline, "svc_prod_v3.joblib")


def download_data():
    # check if mnist data is already downloaded
    mnist_train_set_path = Path("data", "mnist_train_set.csv")
    mnist_test_set_path = Path("data", "mnist_test_set.csv")
    augmented_train_X_set_path = Path("data", "augmented_train_X.csv")
    augmented_train_Y_set_path = Path( "data", "augmented_train_Y.csv")

    if not mnist_train_set_path.exists():
        print("Downloading MNIST training data...")
        # download train set
        file_id = "1Rho1umzwBQTodR7sXVCdUZsJE7xq9EmM"
        download_from_google_drive(file_id, str(mnist_train_set_path))

    if not mnist_test_set_path.exists():
        print("Downloading MNIST test data...")
        # download test set
        file_id = "1qxd-M96DJpYXHfO8xf_XKdDHr3o0xMUE"
        download_from_google_drive(file_id, str(mnist_test_set_path))

    if not augmented_train_X_set_path.exists():
        print("Downloading augmented train X data...")
        # download train set
        file_id = "10TExQfMfM-ku45L9F1LFU4LxobPM7OrR"
        download_from_google_drive(file_id, str(augmented_train_X_set_path))

    if not augmented_train_Y_set_path.exists():
        print("Downloading augmented train Y data...")
        # download train set
        file_id = "10SL3pSwsyiws9_t6upD3CwilrdtJU6Pg"
        download_from_google_drive(file_id, str(augmented_train_Y_set_path))

    return mnist_train_set_path, mnist_test_set_path, augmented_train_X_set_path, augmented_train_Y_set_path


if __name__ == "__main__":

    # download data
    mnist_train_set_path, mnist_test_set_path, augmented_train_X_set_path, augmented_train_Y_set_path = download_data()

    # access train data
    print("Reading MNIST training data...")
    mnist_train_set = pd.read_csv(mnist_train_set_path)

    # access test data
    print("Reading MNIST test data...")
    mnist_test_set = pd.read_csv(mnist_test_set_path)

    # access augmented train data
    print("Reading augmented train data...")
    augmented_train_X = pd.read_csv(augmented_train_X_set_path)
    augmented_train_Y = pd.read_csv(augmented_train_Y_set_path)

    # separate features and target
    # Split training features and target into separate dataset
    train_X = mnist_train_set.drop("class", axis=1)
    train_Y = mnist_train_set["class"]

    # split test features and target into separate dataset
    test_X = mnist_test_set.drop("class", axis=1)
    test_Y = mnist_test_set["class"]

    # # train the model
    # print("Intializing SVC using the best params and regular data...")
    # train_svc(train_X, train_Y, model_name="svc_prod_v3.joblib")

    # # train the model
    # print("Intializing SVC using the best params and regular data...")
    # train_svc(train_X, train_Y)

    # train the model
    print("Intializing SVC using the best params and augmented data...")
    train_svc(augmented_train_X, augmented_train_Y, model_name="svc_prod.joblib")

from api.utils.preprocessing import preprocess_data
from api.utils.common import save_model, download_from_google_drive
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pathlib import Path

import pandas as pd


# helper function to train the Support Vector Classifier (SVC) model
def train_svc(features, target):
    """ Train a Support Vector Classifier (SVC) model on the input data.
    """
    # Best Parameters: {'svc__C': 10, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'}
    # initialize SVC
    print("Intializing SVC...")
    svc = SVC(probability=True, random_state=42, C=10, gamma="scale",
              kernel="rbf")  # Enable probability for AUC

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
    save_model(svc_pipeline, "svc_prod_v1.joblib")


if __name__ == "__main__":
    # check if mnist data is already downloaded
    mnist_train_set_path = Path("api", "data", "mnist_train_set.csv")
    mnist_test_set_path = Path("api", "data", "mnist_test_set.csv")

    if not mnist_train_set_path.exists():
        print("Downloading MNIST training data...")
        # download train set
        file_id = "1Rho1umzwBQTodR7sXVCdUZsJE7xq9EmM"
        # data_dir = str(Path("..", "data", "mnist_train_set.csv"))
        download_from_google_drive(file_id, str(mnist_train_set_path))

    if not mnist_test_set_path.exists():
        print("Downloading MNIST test data...")
        # download test set
        file_id = "1qxd-M96DJpYXHfO8xf_XKdDHr3o0xMUE"
        # data_dir = str(Path("..", "data", "mnist_test_set.csv"))
        download_from_google_drive(file_id, str(mnist_test_set_path))

    # access train data
    print("Reading MNIST training data...")
    mnist_train_set = pd.read_csv(mnist_train_set_path)

    # access test data
    print("Reading MNIST test data...")
    mnist_test_set = pd.read_csv(mnist_test_set_path)
    # separate features and target
    # Split training features and target into separate dataset
    train_X = mnist_train_set.drop("class", axis=1)
    train_Y = mnist_train_set["class"]

    # split test features and target into separate dataset
    test_X = mnist_test_set.drop("class", axis=1)
    test_Y = mnist_test_set["class"]

    # train the model
    train_svc(train_X, train_Y)

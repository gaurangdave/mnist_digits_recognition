# MNIST Digits Recognition

## 📌 Project Goal

The goal of this project is to build an **end-to-end application** for recognizing handwritten digits using the MNIST dataset.

Key Features:
1. **Model Training** : A machine learning model is trained to recognize digits (0-9) from the MNIST dataset.
2. **Interactive Web Application** :
	* Users can draw a handwritten digit using a canvas or upload an image.
	* The application predicts the digit based on the trained model.

## 🎯 Solution Details

### 🧮 Performance Measure
To evaluate and compare models, the following metrics were used:
- **Weighted F1 Score**: To balance precision and recall.
- **Accuracy**: To measure the percentage of correct predictions.
- **Class-wise F1 Score**: To analyze performance for individual classes.
- **Confusion Matrix**: To identify specific misclassifications and analyze errors. 

### 🛠 Data Transformation
The MNIST dataset was preprocessed using **Normalization** to scale pixel intensity values between 0 and 1.

### 📂 Dataset
The MNIST dataset is directly accessible from `sklearn`:
```
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", as_frame=False)
```
* The `train` and `test` set used for training and validation, as well as augmented data set can be accessed from public `Google Drive` folder [here](https://drive.google.com/drive/folders/10FmschultsicypMnWv-uI957F35P2ro5).

### 📒  Notebooks
* [00_get_data.ipynb](https://github.com/gaurangdave/mnist_digits_recognition/blob/main/notebooks/00_get_data.ipynb) Download the dataset and save locally. <a href="https://colab.research.google.com/github/gaurangdave/mnist_digits_recognition/blob/main/notebooks/00_get_data.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

* [01_explore_data.ipynb](https://github.com/gaurangdave/mnist_digits_recognition/blob/main/notebooks/01_explore_data.ipynb) Data exploration and splitting the train/test set.<a href="https://colab.research.google.com/github/gaurangdave/mnist_digits_recognition/blob/main/notebooks/01_explore_data.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

* [02_training_evaluation_colab.ipynb](https://github.com/gaurangdave/mnist_digits_recognition/blob/main/notebooks/02_training_evaluation_colab.ipynb) Model training and evaluation on Google Colab. <a href="https://colab.research.google.com/github/gaurangdave/mnist_digits_recognition/blob/main/notebooks/02_transform_data.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

* [03_training_evaluation_local.ipynb](https://github.com/gaurangdave/mnist_digits_recognition/blob/main/notebooks/03_training_evaluation_local.ipynb) Model training and evaluation on a local machine.

### 🧠 Model Insights
Production Model:
* **Algorithm**: Support Vector Classifier (SVC)
* **Performance**:
  * Weighted F1 Score: **0.98**
  * Accuracy: **0.98**
  * Class-wise average F1 Score: **0.98**
* **Hyperparameters**:
	* kernel: rbf
	* C: 10
	* gamma: scale
	* Trained using normalized data.

**Observations:**
* The model performs well overall but struggles with specific misclassifications:
* 53% of misclassified 6s are predicted as 0.
* 54% of misclassified 4s are predicted as 9.
* Digits like 3, 5, and 8 also have higher misclassification rates, likely due to their visual similarity.
 
* All the trained models including the production one can be accessed in this public `Google Drive` [folder](https://drive.google.com/drive/folders/10GxWYi3NkoZv1Zba9yQ2y88TRM5-xKJG). 

## 💻 Tech Stack

![Environment](https://img.shields.io/badge/Environment-Google_Colab-FCC624?logo=googlecolab&style=for-the-badge)  
![Python](https://img.shields.io/badge/Python-3.12.2-FFD43B?logo=Python&logoColor=blue&style=for-the-badge)  
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-2C2D72?logo=Pandas&logoColor=2C2D72&style=for-the-badge)  
![Plotly](https://img.shields.io/badge/Plotly-5.24.1-239120?logo=Plotly&logoColor=239120&style=for-the-badge)  
![Scikit Learn](https://img.shields.io/badge/scikit_learn-1.5.1-F7931E?logo=scikit-learn&logoColor=F7931E&style=for-the-badge)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-FF6F00?logo=TensorFlow&logoColor=FF6F00&style=for-the-badge)  
![Google Colab](https://img.shields.io/badge/Notebook-Google_Colab-FCC624?logo=googlecolab&style=for-the-badge)  
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-109989?logo=Fastapi&logoColor=109989&style=for-the-badge)  

### Tools and Platforms:
1. **Google Colab**: Used for data exploration, training, and evaluation of the digit recognition model.  
2. **Python**: Primary programming language for machine learning, data preprocessing, and API development.  
3. **Scikit-learn**: For data preprocessing and building classical ML models.  
4. **TensorFlow**: For building and training deep learning models (if needed for improved performance).  
5. **Plotly**: For creating interactive visualizations during the analysis phase.  
6. **FastAPI**: For exposing the trained model as an API and building backend services.  


## Setting Up Secrets in Google Colab

* To run this project on Google Colab securely, you will need to set up a secret to store the shared Google Drive folder path. 
* This ensures your intermediate data storage path is not exposed when sharing notebooks.

### Step-by-Step Instructions:

1. Open Google Colab:
* Open the notebook you want to run in Google Colab.
2. Access the Secrets Manager:
* In the Colab notebook, click on the lock icon 🔒 in the top-right corner of the interface (next to the table of contents icon).
* This will open the Secrets Manager.
3. Add a New Secret:
* Click “Add new secret”.
* Set the name of the secret as: SHARED_DRIVE_PATH.
* Set the value of the secret as the path to your shared Google Drive folder. For example:
```bash
/content/drive/My Drive/shared_folder_name
```
4. Access the Secret in Code:
* The secret is accessed in the notebook using the following code:

```bash
from google.colab import userdata

# Retrieve the Google Drive path stored in secrets
shared_folder_path = userdata.get("SHARED_DRIVE_PATH")

# Use the shared folder path in your code
print(f"Using shared folder: {shared_folder_path}")
```
5. Ensure Google Drive is Mounted:
* Mount Google Drive in your notebook before using the secret:

```bash
from google.colab import drive
drive.mount('/content/drive')
```

6. Verify the Path:
* Check if the folder exists:

```bash
import os
if os.path.exists(shared_folder_path):
    print("Shared folder is accessible!")
else:
    print("Shared folder path is incorrect. Please check your secret.")
```
Important Notes:

* Secrets are private and not shared with others, even if you share your Colab notebook.
* Each user must add their own `SHARED_DRIVE_PATH` secret to run the code successfully.
* If you’re collaborating with others, ensure everyone uses a consistent folder structure in their Google Drive.


## 💻 Running Locally

### Install Dependencies

- Create conda environment with `Python 3.12`

```bash
  conda create -n ml python=3.12
```

- Activate the environment

```bash
  conda activate ml
```

- Install ML Libraries

```bash
conda install numpy pandas scikit-learn matplotlib seaborn plotly jupyter ipykernel -y
```

```bash
conda install -c conda-forge python-kaleido
```

- Install GDown
```bash
conda install conda-forge::gdown
```

- Install DotEnv
```bash
conda install conda-forge::python-dotenv
```

- Install FastAPI

```bash
conda install -c conda-forge fastapi uvicorn -y
```
### Training Model for API

* Run the following command to train model(s) for production use

```bash
python -m scripts.training
```

### Running the API
* Run the following command to start the API server

```bash
uvicorn api.main:app --reload
```

* Go to the following URL to access API Docs
```URL
http://localhost:8000/docs
```

### Running API Test Cases
* Run the following command to run all the test cases
```bash
pytest
```

* Run the following command to run a specific test case
  * Here `test_predict_digit_endpoint` is the name of test case in `tests/test_api.py` file.
```bash
pytest -k test_predict_digit_endpoint
```

### API Reference

| Action                                           | HTTP Method | Endpoint                                 |
|--------------------------------------------------|-------------|------------------------------------------|
| Get predictions using                            | **`POST`**  | `/predict`                               |

### Accessing UI

* To access the UI application, simply go to `http://localhost:8000` after starting the server and you should be able to access the webapp. 

### App Screenshots

#### Default Home Page
![alt text](image.png)

#### Prediction
![alt text](image-1.png)

## Visualizations

### Metric Comparison Of Various Models
![image](https://github.com/user-attachments/assets/e1f295d4-3dcf-4aa0-b5f3-dd30a1f61186)

### Confusion Matrix for Production Model against test data
![test_validation](https://github.com/user-attachments/assets/5ff76a3b-9bf8-40bb-8582-3fefdcfd3170)

### Confusion Matrix specifying errors of Production Model against test data
![test_validation_errors](https://github.com/user-attachments/assets/4bb9f081-df3b-4b0e-8b2d-beea41ccb041)

## Project Insights
* The final model achieved a `Weighted F1 Score` of `0.98` and `Accuracy` of `0.98` 
* We were also able to acheive an average classbased `F1` score of `0.98`, which means our model has a balanced `Precision` and `Recall`, which means we have high `True Positive` rates and low `False Negative` rates. 
* Our model still makes errors in some classification, 
  * e.g. `53%` of misclassified 6 is classified as 0 and `54%` of misclassified 4s are classified as 9s. 
  * Similarly there is higher rate of misclassification between 3, 5 and 8 and hypothesis is this could be because of similarity of these digits in hand written format. 

### Next Steps
1.	Train the model with augmented data to improve performance.
2.	Explore deep learning models like CNNs for potentially better accuracy.

## Lessons Learnt
* Improved understanding of **classification algorithms**, metrics, and error analysis.
* Learned data exploration techniques for image data.
* Debugged and resolved a production issue with the normalization pipeline, gaining valuable insight into E2E system testing.
* Familiarity with the **kernel trick** and its role in handling linear vs non-linear data.

## 🌟 Project Highlights
1. End-to-End Application:
* Includes a model training pipeline, API, and a front-end interface.
2. User Interaction:
* Users can draw digits on a canvas and get predictions in real time.

## 🚀 About Me

A jack of all trades in software engineering, with 15 years of crafting full-stack solutions, scalable architectures, and pixel-perfect designs. Now expanding my horizons into AI/ML, blending experience with curiosity to build the future of tech—one model at a time.

## 🔗 Links

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://gaurangdave.me/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gaurangvdave/)

## 🛠 Skills

`Python`, `Jupyter Notebook`, `scikit-learn`, `FastAPI`, `Plotly`, `Conda`

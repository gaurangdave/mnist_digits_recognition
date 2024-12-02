# MNIST Digits Recognition

## Project Goal

The goal of this project is to build an end-to-end application that recognizes handwritten digits using the MNIST dataset. The application will allow users to draw digits (or provide handwritten digit images) and predict the digit using a trained machine learning model. This involves:
1. Training a machine learning model to recognize digits (0-9) from the MNIST dataset.
1. Creating a user-friendly web application where:
	* Users can input a handwritten digit using a canvas or upload an image.
	* The application predicts the digit based on the trained model.

## Solution Details

### Performance Measure

### Data Transformation

### Dataset

### Notebooks

### Models

## Tech Stack

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

## Installation (For Local Development)

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
## Running the API
* Run the following command to start the API server

```bash
uvicorn api.main:app --reload
```

* Go to the following URL to access API Docs
```URL
http://localhost:8000/docs
```

## API Reference

## Visualizations

## Project Insights

### Next Steps

## Lessons Learnt

## 🚀 About Me

A jack of all trades in software engineering, with 15 years of crafting full-stack solutions, scalable architectures, and pixel-perfect designs. Now expanding my horizons into AI/ML, blending experience with curiosity to build the future of tech—one model at a time.

## 🔗 Links

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://gaurangdave.me/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gaurangvdave/)

## 🛠 Skills

`Python`, `Jupyter Notebook`, `scikit-learn`, `FastAPI`, `Plotly`, `Conda`

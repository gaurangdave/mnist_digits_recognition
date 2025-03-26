import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Get the HuggingFace API key from the environment
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HuggingFace API key not found. Please set HF_API_KEY in your .env file.")

# Authenticate with HuggingFace API
api = HfApi()
user = api.whoami()
user_name = user['name']
print(f"Logged in as {user_name}")
print("Successfully authenticated with HuggingFace API.")

## hugging face repo id
model_name = "mnist_digits_recognition"
repo_id = f"{user_name}/{model_name}"


## download models
model_root = "models"
os.makedirs(model_root, exist_ok=True)

all_models = api.list_repo_files(repo_id, repo_type="model")

print(f"Models in {repo_id}:")
for model in all_models:
    ## download to tmp folder
    file_path = hf_hub_download(repo_id=repo_id, filename=model, token=HF_TOKEN, local_dir=model_root, repo_type="model")
    ## move to model folder
    os.rename(file_path, os.path.join(model_root, model))
    print(f"Downloaded {model} to {file_path}")

print(f"✅ All files downloaded to {model_root}")

import zipfile
import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# Folder to save the downloaded zip (only filenames needed)
DATASET_DIR = "ravdess_filenames"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

print("Connecting to Kaggle...")
api = KaggleApi()
api.authenticate()

print("Downloading dataset metadata ONLY...")

# Download zip
api.dataset_download_files("uwrfkaggler/ravdess-emotional-speech-audio", path=DATASET_DIR, unzip=True)

print("\nDownload complete!")
print("Extracted folder:", DATASET_DIR)

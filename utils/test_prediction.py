import requests
import os
import json

TEST_FILE = "venv/lib/python3.11/site-packages/scipy/io/tests/data/test-44100Hz-le-1ch-4bytes.wav"
URL = "http://127.0.0.1:5001/predict_emotion"

if not os.path.exists(TEST_FILE):
    print(f"Error: File {TEST_FILE} not found")
    exit(1)

files = {'file': open(TEST_FILE, 'rb')}
try:
    response = requests.post(URL, files=files)
    print("Status Code:", response.status_code)
    data = response.json()
    print("Keys found:", data.keys())
    if "probabilities" in data:
        print("Probabilities found!")
        print(json.dumps(data["probabilities"], indent=2))
    else:
        print("ERROR: 'probabilities' key missing")
        
except Exception as e:
    print("Request failed:", e)

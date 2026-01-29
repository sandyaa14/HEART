import pickle
import numpy as np
import os

PKL_PATH = "Emotion_detection/ravdess_train.pkl"

if not os.path.exists(PKL_PATH):
    print("PKL file not found")
    exit()

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

X_train = np.array([item[0] for item in data])

print("Shape:", X_train.shape)
print("Min:", np.min(X_train))
print("Max:", np.max(X_train))
print("Mean:", np.mean(X_train))
print("Std:", np.std(X_train))

# Check first feature vs last feature stats
print("Feat 0 stats: mean=%.3f, std=%.3f" % (np.mean(X_train[:, 0]), np.std(X_train[:, 0])))
print("Feat 100 stats: mean=%.3f, std=%.3f" % (np.mean(X_train[:, 100]), np.std(X_train[:, 100])))

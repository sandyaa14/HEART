# 04_deep_ensemble_meta_model.py

# ==============================
# Deep Stacking Ensemble with Logistic Regression Meta-Model
# ==============================

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import joblib  # to save the meta model

# ------------------------------
# Local Paths (Windows)
# ------------------------------
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(SCRIPT_DIR, "Emotion_detection")
MODEL_SAVE_FOLDER = os.path.join(SAVE_FOLDER, "Models")

os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)

print("Loading data from:", SAVE_FOLDER)
print("Loading models from:", MODEL_SAVE_FOLDER)

# ------------------------------
# Load Validation & Test Data
# ------------------------------
try:
    with open(os.path.join(SAVE_FOLDER, 'ravdess_val.pkl'), 'rb') as f:
        val_data = pickle.load(f)

    with open(os.path.join(SAVE_FOLDER, 'ravdess_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    X_val = np.array([item[0] for item in val_data])
    y_val_labels = np.array([item[1] for item in val_data])

    X_test = np.array([item[0] for item in test_data])
    y_test_labels = np.array([item[1] for item in test_data])

    print("Validation and Test data loaded successfully.")
    print(f"X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")

except Exception as e:
    print(f"[ERROR] Data loading error: {e}")
    exit()

# ------------------------------
# Encode Labels to integers
# ------------------------------
target_emotions = ['neutral','happy','sad','angry','fearful','disgust','surprised','calm']
label_to_index = {emotion: i for i, emotion in enumerate(target_emotions)}

y_val_int = np.array([label_to_index[e] for e in y_val_labels])
y_test_int = np.array([label_to_index[e] for e in y_test_labels])

print("Labels encoded successfully.")

# ------------------------------
# Load Base Models
# ------------------------------
multi_model_path = os.path.join(MODEL_SAVE_FOLDER, "multi_dnn_model.h5")
if os.path.exists(multi_model_path):
    multi_model = load_model(multi_model_path)
    print("Multi-class DNN loaded.")
else:
    print("[ERROR] Multi-class model not found!")
    exit()

channel_models = {}
for emotion in target_emotions:
    path = os.path.join(MODEL_SAVE_FOLDER, f"{emotion}_model.h5")
    if os.path.exists(path):
        channel_models[emotion] = load_model(path)
        print(f"Binary model for '{emotion}' loaded.")
    else:
        channel_models[emotion] = None
        print(f"[WARNING] Missing model for '{emotion}' (will skip this channel).")

# ------------------------------
# Build Meta Features (no caching; rebuild fresh)
# ------------------------------
print("\n===== Building Meta Features for Logistic Regression =====")

meta_features_val = []
meta_features_test = []

# Multi-class model predictions (8 probs)
multi_val = multi_model.predict(X_val, verbose=0)
multi_test = multi_model.predict(X_test, verbose=0)
meta_features_val.append(multi_val)
meta_features_test.append(multi_test)

# Binary model predictions (1 prob per model)
for emotion, model in channel_models.items():
    if model is not None:
        pred_val = model.predict(X_val, verbose=0).flatten().reshape(-1, 1)
        pred_test = model.predict(X_test, verbose=0).flatten().reshape(-1, 1)
        meta_features_val.append(pred_val)
        meta_features_test.append(pred_test)

# Stack horizontally: shape (n_samples, n_meta_features)
meta_features_val = np.hstack(meta_features_val)
meta_features_test = np.hstack(meta_features_test)

print(f"Meta features shapes → val: {meta_features_val.shape}, test: {meta_features_test.shape}")

# ------------------------------
# Train Logistic Regression Meta-Model
# ------------------------------
print("\n===== Training Logistic Regression Meta-Model =====")

meta_clf = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs'
)

meta_clf.fit(meta_features_val, y_val_int)

# Save the meta model
meta_clf_path = os.path.join(MODEL_SAVE_FOLDER, "meta_logreg_model.joblib")
joblib.dump(meta_clf, meta_clf_path)
print(f"Meta Logistic Regression model saved at: {meta_clf_path}")

# ------------------------------
# Evaluate Meta-Model
# ------------------------------
print("\n===== Evaluating Logistic Regression Meta-Model =====")

val_pred = meta_clf.predict(meta_features_val)
test_pred = meta_clf.predict(meta_features_test)

val_acc = accuracy_score(y_val_int, val_pred) * 100
test_acc = accuracy_score(y_test_int, test_pred) * 100

print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

print("\nClassification Report on Test Data:")
print(classification_report(y_test_int, test_pred, target_names=target_emotions))

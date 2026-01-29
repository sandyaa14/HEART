# train_gender_model_ravdess.py
"""
Train Gender Classification Model using EXISTING RAVDESS Dataset
Extracts gender from RAVDESS actor IDs in the pickle files
Actor IDs: Odd = Male, Even = Female
"""

import os
import numpy as np
from gender_features import extract_gender_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import pickle
import librosa

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAVDESS_DATA_DIR = os.path.join(SCRIPT_DIR, "Emotion_detection")
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print("=" * 70)
print("Gender Classification Model Training - RAVDESS Dataset")
print("=" * 70)


def get_gender_from_emotion_data():
    """
    Extract gender features from the RAVDESS pickle files.
    The pickle files contain (feature_vector, emotion_label) tuples.
    We need to re-extract features with our gender feature extractor
    and infer gender from file patterns.
    
    Since we don't have actor IDs in the pickle, we'll need to find
    the raw audio files to extract gender properly.
    """
    print("\n[1/5] Looking for RAVDESS audio files...")
    
    # Common locations where RAVDESS might be
    possible_dirs = [
        os.path.join(SCRIPT_DIR, "datasets", "ravdess"),
        os.path.join(SCRIPT_DIR, "Emotion_detection", "ravdess"),
        os.path.expanduser("~/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio"),
    ]
    
    audio_files = []
    print(f"  Searching in {len(possible_dirs)} locations...")
    
    for base_dir in possible_dirs:
        if os.path.exists(base_dir):
            print(f"  ✅ Found directory: {base_dir}")
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('.wav') and 'Actor_' in root:
                        audio_files.append(os.path.join(root, file))
            
            if audio_files:
                print(f"  Found {len(audio_files)} wav files in {base_dir}")
                break
        else:
            print(f"  ❌ Not found: {base_dir}")
    
    if not audio_files:
        print("\n❌ Could not find RAVDESS audio files")
        print("   Searched in:")
        for d in possible_dirs:
            print(f"     - {d}")
        print("\n💡 Please download RAVDESS dataset from:")
        print("   https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio")
        return None, None
    
    print(f"\n✓ Found {len(audio_files)} audio files")
    
    # Extract features and labels
    print("\n[2/5] Extracting gender features from audio...")
    features_list = []
    labels_list = []
    
    for i, audio_path in enumerate(audio_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(audio_files)} files...")
        
        try:
            # Extract actor ID from path (Actor_XX folder)
            actor_folder = [part for part in audio_path.split(os.sep) if 'Actor_' in part]
            if not actor_folder:
                continue
            
            actor_id = int(actor_folder[0].split('_')[1])
            gender = 'male' if actor_id % 2 == 1 else 'female'
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=48000, mono=True)
            
            # Extract features
            features = extract_gender_features(audio, sr)
            
            features_list.append(features)
            labels_list.append(gender)
            
        except Exception as e:
            print(f"  Warning: Failed to process {audio_path}: {e}")
            continue
    
    if len(features_list) == 0:
        print("\n❌ No features extracted")
        return None, None
    
    X = np.vstack(features_list)
    y = np.array(labels_list)
    
    print(f"\n✓ Extracted features from {len(y)} samples")
    print(f"  Males: {np.sum(y == 'male')}")
    print(f"  Females: {np.sum(y == 'female')}")
    
    return X, y


# Main training pipeline
print("\nStarting training pipeline...")

# Load data from RAVDESS
X, y = get_gender_from_emotion_data()

if X is None:
    print("\n❌ Training aborted - no data available")
    print("   Please ensure RAVDESS dataset is downloaded")
    exit(1)

# Split data
print("\n[3/5] Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"  Train: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

# Scale features
print("\n[4/5] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n[5/5] Training models...")

# Model 1: SVM
print("\n  Training SVM...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

svm_val_acc = accuracy_score(y_val, svm_model.predict(X_val_scaled))
svm_test_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))
print(f"    Validation Accuracy: {svm_val_acc * 100:.2f}%")
print(f"    Test Accuracy: {svm_test_acc * 100:.2f}%")

# Model 2: Random Forest
print("\n  Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

rf_val_acc = accuracy_score(y_val, rf_model.predict(X_val_scaled))
rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))
print(f"    Validation Accuracy: {rf_val_acc * 100:.2f}%")
print(f"    Test Accuracy: {rf_test_acc * 100:.2f}%")

# Select best model
print("\n[6/6] Selecting best model...")
if svm_test_acc >= rf_test_acc:
    best_model = svm_model
    best_name = "SVM"
    best_acc = svm_test_acc
else:
    best_model = rf_model
    best_name = "RandomForest"
    best_acc = rf_test_acc

print(f"  Best model: {best_name} ({best_acc * 100:.2f}% accuracy)")

# Detailed evaluation
print("\n" + "=" * 70)
print(f"FINAL EVALUATION - {best_name}")
print("=" * 70)

y_pred = best_model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['female', 'male']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n  [[Female->Female, Female->Male]")
print(f"   [Male->Female,   Male->Male]]")

# Save model and scaler
model_path = os.path.join(MODEL_SAVE_DIR, "gender_classifier.pkl")
scaler_path = os.path.join(MODEL_SAVE_DIR, "gender_scaler.pkl")

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✓ Model saved to: {model_path}")
print(f"✓ Scaler saved to: {scaler_path}")

print("\n" + "=" * 70)
print("✅ Training complete!")
print(f"   Final Test Accuracy: {best_acc * 100:.2f}%")
print("=" * 70)

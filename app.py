from flask import Flask, request, jsonify, render_template, send_from_directory
import io
import os
import numpy as np
import soundfile as sf
import librosa
from flask_cors import CORS
import joblib
from sklearn.ensemble import RandomForestClassifier
import json
import time
from datetime import datetime

# Try to import TensorFlow/Keras - optional for basic functionality
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available - emotion detection will be disabled")

# =========================================================
# Path constants
# =========================================================
MODEL_PATH = "ravdess_emotion_model.pkl"
TRAIN_PKL_PATHS = [
    "ravdess_train.pkl",
    "Emotion_detection/ravdess_train.pkl",
    "C:/Final Year/Emotion_detection/ravdess_train.pkl",
]

EMOTION_MODEL_PATH = "Emotion_detection/Models/"
UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE = "static/history.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

# =========================================================
# History Utilities
# =========================================================
def save_recording(audio_bytes, suffix="rec"):
    """Saves audio bytes to disk and returns the filename."""
    timestamp = int(time.time())
    filename = f"{suffix}_{timestamp}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    return filename

def log_history(filename, emotion, gender, confidence):
    """Appends entry to history.json"""
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except:
        history = []
    
    # Clean filename just in case
    clean_filename = filename.split('/')[-1]

    entry = {
        "id": str(int(time.time())),
        "filename": f"uploads/{clean_filename}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": emotion,
        "gender": gender,
        "confidence": float(confidence)
    }
    
    # Prepend
    history.insert(0, entry)
    # Keep last 50
    history = history[:50]
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# =========================================================
# Emotion labels (ORDER MUST MATCH TRAINING)
# =========================================================
# CRITICAL: This order MUST match the training script (04_deep_ensemble_meta_model.py line 55)
EMOTION_LABELS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
    "calm",
]

# =========================================================
# Load Deep Learning Emotion Models
# =========================================================
emotion_models = {}
multi_model = None
meta_model_logreg = None

if TF_AVAILABLE and os.path.exists(EMOTION_MODEL_PATH):
    try:
        # 1. Load Binary Models
        emotion_models = {
            "angry": tf.keras.models.load_model(EMOTION_MODEL_PATH + "angry_model.h5"),
            "calm": tf.keras.models.load_model(EMOTION_MODEL_PATH + "calm_model.h5"),
            "disgust": tf.keras.models.load_model(
                EMOTION_MODEL_PATH + "disgust_model.h5"
            ),
            "fearful": tf.keras.models.load_model(
                EMOTION_MODEL_PATH + "fearful_model.h5"
            ),
            "happy": tf.keras.models.load_model(EMOTION_MODEL_PATH + "happy_model.h5"),
            "neutral": tf.keras.models.load_model(
                EMOTION_MODEL_PATH + "neutral_model.h5"
            ),
            "sad": tf.keras.models.load_model(EMOTION_MODEL_PATH + "sad_model.h5"),
            "surprised": tf.keras.models.load_model(
                EMOTION_MODEL_PATH + "surprised_model.h5"
            ),
        }

        # 2. Load Multi-class DNN Model
        multi_model_path = EMOTION_MODEL_PATH + "multi_dnn_model.h5"
        if os.path.exists(multi_model_path):
            multi_model = tf.keras.models.load_model(multi_model_path)
            print("✅ Multi-class DNN model loaded")

        # 3. Load Meta Model (Logistic Regression)
        meta_logreg_path = EMOTION_MODEL_PATH + "meta_logreg_model.joblib"
        if os.path.exists(meta_logreg_path):
            meta_model_logreg = joblib.load(meta_logreg_path)
            print("✅ Meta Logistic Regression model loaded")

        print("✅ Deep Learning emotion models loaded")

    except Exception as e:
        print("⚠️ DL emotion model loading failed:", e)
else:
    print("ℹ️ DL emotion models not found or TensorFlow not available")

# =========================================================
# Flask App
# =========================================================
app = Flask(__name__)
CORS(app)


# =========================================================
# Utility Functions
# =========================================================
def load_wav_from_bytes(b):
    bio = io.BytesIO(b)
    try:
        audio, sr = sf.read(bio)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        # 1. Trim Silence (remove start/end silence)
        audio, _ = librosa.effects.trim(audio, top_db=30)

        # 2. Normalize (Max Amplitude = 1.0)
        # This fixes the "low energy" issue for quiet user mics
        if np.max(np.abs(audio)) > 0:
            audio = librosa.util.normalize(audio)
            
        return audio, sr
    except Exception:
        bio.seek(0)
        audio, sr = librosa.load(bio, sr=None, mono=True)
        # Apply same processing to fallback
        audio, _ = librosa.effects.trim(audio, top_db=30)
        if np.max(np.abs(audio)) > 0:
            audio = librosa.util.normalize(audio)
        return audio.astype(np.float32), sr


# =========================================================
# Feature extraction for DL emotion models (188 features)
# CRITICAL: Must match Feature_Extraction script EXACTLY
# =========================================================
def extract_emotion_features(audio, sr):
    """
    Extract features matching the training script exactly.
    Training uses: 50 MFCCs, 128 mel bins, ZCR, spectral contrast, pitch, energy
    at 48kHz sample rate.
    """
    audio = audio.astype(np.float64)
    
    # Resample to 48000 Hz to match training
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        sr = 48000
    
    # Extract features matching Feature_Extraction script
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=50)  # Changed from 40 to 50
    mfcc_mean = np.mean(mfcc, axis=1)
    
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_mean = np.mean(mel, axis=1)
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=1)
    
    pitches, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitches)
    
    energy = np.sum(audio ** 2) / len(audio)
    
    features = np.concatenate([
        mfcc_mean,           # 50 features
        mel_mean,            # 128 features
        [zcr],               # 1 feature
        spectral_contrast,   # 7 features (default)
        [pitch_mean, energy] # 2 features
    ]).astype(np.float32)
    
    # Categorized details for explainability
    def safe_float(x):
        try:
            val = float(x)
            return round(val, 4) if np.isfinite(val) else 0.0
        except:
            return 0.0

    details = {
        "mfcc": [safe_float(x) for x in mfcc_mean],
        "mel": [safe_float(x) for x in mel_mean],
        "zcr": safe_float(zcr),
        "spectral_contrast": [safe_float(x) for x in spectral_contrast],
        "pitch": safe_float(pitch_mean),
        "energy": safe_float(energy)
    }
    
    return features, details


# =========================================================
# CORRECT Deep Learning Emotion Prediction
# =========================================================
def predict_emotion(audio, sr):
    """
    Handles sigmoid-based binary emotion models correctly
    and feeds 16 features into the meta model.
    """
    emotion, confidence, probs, model_used, feature_details = predict_emotion_with_confidence(audio, sr)
    return emotion


def predict_emotion_with_confidence(audio, sr):
    """
    Uses Ensemble Logic from 04_deep_ensemble_meta_model.py:
    1. Multi-class DNN predictions (8 probs)
    2. Binary Emotion Models (1 prob each -> 8 probs)
    Total Meta Features: 16
    """

    # 1. Feature extraction
    features, feature_details = extract_emotion_features(audio, sr)
    features = np.expand_dims(features, axis=0)  # (1, 188)
    
    # ---------------------------------------------------------
    # PART A: Multi-class DNN Predictions
    # ---------------------------------------------------------
    if multi_model is None:
        print("ERROR: Multi-class model not loaded")
        return "error", 0.0, {}, "Error", feature_details

    multi_val = multi_model.predict(features, verbose=0) # Shape (1, 8)
    
    # ---------------------------------------------------------
    # PART B: Binary Model Predictions
    # ---------------------------------------------------------
    binary_preds = []
    
    # Order matters! Must match 04_deep_ensemble_meta_model.py
    target_emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'calm']
    
    for emotion in target_emotions:
        if emotion not in emotion_models:
             # print(f"DEBUG: Missing model for {emotion}")
             binary_preds.append(0.0) # Fallback
             continue
             
        model = emotion_models[emotion]
        probs = model.predict(features, verbose=0)

        # Handle Sigmoid output (shape can be (1,) or (1,1))
        if len(probs.shape) == 1 or (len(probs.shape) == 2 and probs.shape[1] == 1):
            # Sigmoid: single probability value
            p = float(probs.flatten()[0])
            binary_preds.append(p)
        else:
            # Softmax: [p_not, p_yes]
            p_yes = float(probs[0][1])
            binary_preds.append(p_yes)

    binary_preds = np.array(binary_preds).reshape(1, -1) # Shape (1, 8)
    
    # ---------------------------------------------------------
    # PART C: Construct Meta Features
    # ---------------------------------------------------------
    # Stack horizontally: [Multi_Probs (8) | Binary_Probs (8)] -> Total 16
    meta_inputs = np.hstack([multi_val, binary_preds])

    # ---------------------------------------------------------
    # PART D: Meta Model Prediction & Ensemble Logic
    # ---------------------------------------------------------
    if meta_model_logreg is None:
        print("ERROR: Meta Logistic Regression model not loaded")
        return "error", 0.0, {}, "Error", feature_details

    final_pred_probs = meta_model_logreg.predict_proba(meta_inputs)
    final_class_idx = np.argmax(final_pred_probs)
    
    dl_confidence = float(final_pred_probs[0][final_class_idx])
    dl_emotion = target_emotions[final_class_idx]
    
    # Create probability dictionary for DL model
    probabilities = {
        emotion: float(prob) 
        for emotion, prob in zip(target_emotions, final_pred_probs[0])
    }

    # ---------------------------------------------------------
    # PART E: Random Forest Fallback / Ensemble
    # ---------------------------------------------------------
    # Using the pre-loaded Random Forest model 'emotion_model'
    # Default to DL result first
    final_emotion = dl_emotion
    final_confidence = dl_confidence
    model_used = "DeepLearning"

    # Strategy: Trust RF if DL is uncertain or predicts 'calm' (known bias)
    if emotion_model is not None:
        try:
            # RF expects (1, 188) features
            rf_probs = emotion_model.predict_proba(features)[0]
            rf_class_idx = np.argmax(rf_probs)
            rf_emotion = emotion_model.classes_[rf_class_idx]
            rf_confidence = float(rf_probs[rf_class_idx])
            
            # Debug log
            print(f"DEBUG: DL({dl_emotion}, {dl_confidence:.2f}) vs RF({rf_emotion}, {rf_confidence:.2f})")

            # Condition 1: If DL is 'calm' and confidence is not extremely high, check RF
            if dl_emotion == 'calm' and dl_confidence < 0.85:
                 if rf_confidence > 0.4:
                     print(f"ℹ️ DL predicted 'calm' (conf {dl_confidence:.2f}). Switching to RF '{rf_emotion}' (conf {rf_confidence:.2f}).")
                     final_emotion = rf_emotion
                     final_confidence = rf_confidence
                     model_used = "RandomForest (Bias Correction)"
                 else:
                     print(f"ℹ️ DL predicting 'calm' (conf {dl_confidence:.2f}). RF '{rf_emotion}' is too low (conf {rf_confidence:.2f}). Keeping DL.")

            # Condition 2: If DL is generally uncertain (< 0.50) and RF is confident
            elif dl_confidence < 0.50 and rf_confidence > dl_confidence and rf_confidence > 0.4:
                 print(f"ℹ️ DL uncertain. Switching to Random Forest '{rf_emotion}'.")
                 final_emotion = rf_emotion
                 final_confidence = rf_confidence
                 model_used = "RandomForest (Fallback)"
            
            # Merge probabilities for UI (optional, but good for radar chart)
            # We will blend them 50/50 if we want a true ensemble, 
            # but for now let's just update the specific chosen emotion's prob to match the choice
            if model_used.startswith("RandomForest"):
                 # Update probs to reflect RF
                  probabilities = {
                    cls: float(prob) 
                    for cls, prob in zip(emotion_model.classes_, rf_probs)
                }

        except Exception as e:
            print(f"⚠️ RF Prediction failed: {e}")

    # 🔒 Final Confidence Check
    if final_confidence < 0.35: 
        return "uncertain", final_confidence, probabilities, "Uncertain", feature_details

    return final_emotion, final_confidence, probabilities, model_used, feature_details


# =========================================================
# ML Emotion Model (RandomForest fallback – untouched)
# =========================================================
def try_load_training_pkl():
    import pickle

    for p in TRAIN_PKL_PATHS:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f), p
    return None, None


def prepare_and_train_model():
    if os.path.exists(MODEL_PATH):
        mdl = joblib.load(MODEL_PATH)
        return mdl, mdl.expected_feature_length_

    data, _ = try_load_training_pkl()
    if data is None:
        return None, None

    X, y = zip(*data)
    X = np.vstack(X)
    y = np.array(y)

    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    clf.expected_feature_length_ = X.shape[1]
    joblib.dump(clf, MODEL_PATH)

    return clf, clf.expected_feature_length_


emotion_model, emotion_feat_len = prepare_and_train_model()


# =========================================================
# Routes
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# Load Gender Model
# =========================================================
GENDER_MODEL_PATH = "models/gender_classifier.pkl"
GENDER_SCALER_PATH = "models/gender_scaler.pkl"
gender_model = None
gender_scaler = None

if os.path.exists(GENDER_MODEL_PATH) and os.path.exists(GENDER_SCALER_PATH):
    try:
        gender_model = joblib.load(GENDER_MODEL_PATH)
        gender_scaler = joblib.load(GENDER_SCALER_PATH)
        print("✅ Gender classification model loaded")
    except Exception as e:
        print(f"⚠️ Gender model loading failed: {e}")
else:
    print("ℹ️ Gender model not found - training required")

from gender_features import extract_gender_features

@app.route("/predict_gender", methods=["POST"])
def predict_gender():
    audio, sr = load_wav_from_bytes(request.files["file"].read())

    # 1. Extract Features
    try:
        features = extract_gender_features(audio, sr)
        features = features.reshape(1, -1)
        
        # 2. Prediction
        if gender_model and gender_scaler:
            features_scaled = gender_scaler.transform(features)
            
            # Predict class and probability
            prediction = gender_model.predict(features_scaled)[0]
            probs = gender_model.predict_proba(features_scaled)[0]
            
            # Get confidence
            class_idx = np.where(gender_model.classes_ == prediction)[0][0]
            confidence = probs[class_idx]
            
            gender = prediction
            
        else:
            # Fallback to simple pitch logic if model fails
            print("⚠️ Using fallback pitch logic for gender")
            f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=sr)
            f0 = f0[~np.isnan(f0)]
            if len(f0) > 0:
                pitch = np.median(f0)
                gender = "male" if pitch < 165 else "female"
                confidence = 0.60 # Low confidence for fallback
            else:
                return jsonify({"gender": "unknown", "confidence": 0.0, "pitch": 0.0})

        # Calculate pitch for display purposes
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=sr)
        f0 = f0[~np.isnan(f0)]
        pitch = np.median(f0) if len(f0) > 0 else 0.0

        print(f"DEBUG: Gender Predict - {gender} ({confidence:.2f})")

        return jsonify(
            {
                "gender": gender,
                "confidence": round(float(confidence), 3),
                "pitch": round(float(pitch), 2),
                "model": "ML Model" if gender_model else "Heuristic (Fallback)"
            }
        )

    except Exception as e:
        print(f"ERROR in gender prediction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_emotion", methods=["POST"])
def predict_emotion_api():
    file_bytes = request.files["file"].read()
    audio, sr = load_wav_from_bytes(file_bytes)
    
    # 0. Check for silence/low volume
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.005:
        print(f"⚠️ Silence detected (RMS: {rms:.4f})")
        return jsonify(
            {
                "emotion": "neutral",
                "confidence": 0.0,
                "probabilities": {},
                "model": "Silence Detector",
                "warning": "Volume too low. Please speak closer to the mic."
            }
        )

    if not emotion_models or meta_model_logreg is None:
        return jsonify({"error": "DL emotion models not loaded"}), 500

    # Get emotion prediction with confidence and model info
    emotion, confidence, probabilities, model_name, feature_details = predict_emotion_with_confidence(audio, sr)
    
    # --- SAVE HISTORY ---
    # We need to save the original bytes (or re-encoded wav)
    fname = save_recording(file_bytes)
    
    # We return the filename so the frontend can trigger the log with full metadata (gender + emotion)
    # log_history(fname, emotion, "--", confidence) <- REMOVED

    print(f"DEBUG: Emotion Predict - {emotion} ({confidence:.2f}) [Model: {model_name}]")

    return jsonify(
        {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "probabilities": probabilities,
            "model": model_name,
            "filename": fname,
            "feature_details": feature_details
        }
    )


@app.route("/get_audio_info", methods=["POST"])
def get_audio_info():
    """Get additional audio information like duration, sample rate, etc."""
    try:
        audio, sr = load_wav_from_bytes(request.files["file"].read())
        duration = len(audio) / sr

        # Calculate some basic audio features
        rms = np.sqrt(np.mean(audio**2))  # Root mean square (volume level)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio)[0])

        return jsonify(
            {
                "duration": round(duration, 2),
                "sample_rate": int(sr),
                "samples": len(audio),
                "rms": round(float(rms), 4),
                "zero_crossing_rate": round(float(zero_crossing_rate), 4),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_history", methods=["GET"])
def get_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify([])


@app.route("/log_history_entry", methods=["POST"])
def log_history_entry():
    data = request.json
    try:
        # We now expect more data
        # filename, emotion, gender, confidence are expanding
        # We will store everything passed in 'details' or just flat
        
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
            
        clean_filename = data.get("filename").replace("uploads/", "").split('/')[-1]
        
        entry = {
            "id": str(int(time.time())),
            "filename": f"uploads/{clean_filename}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "emotion": data.get("emotion"),
            "gender": data.get("gender"),
            "confidence": float(data.get("confidence", 0)),
            # New fields for Lab
            "probabilities": data.get("probabilities", {}),
            "info": data.get("info", {}),
            "feature_details": data.get("feature_details", {})
        }
        
        history.insert(0, entry)
        history = history[:50]
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"Error logging history: {e}")
        return jsonify({"error": str(e)}), 500


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    app.run(debug=True, port=5001)

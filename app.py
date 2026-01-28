from flask import Flask, request, jsonify, render_template
import io
import os
import numpy as np
import soundfile as sf
import librosa
from flask_cors import CORS
import joblib
from sklearn.ensemble import RandomForestClassifier

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

# =========================================================
# Emotion labels (ORDER MUST MATCH TRAINING)
# =========================================================
EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
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
        return audio.astype(np.float32), sr
    except Exception:
        bio.seek(0)
        audio, sr = librosa.load(bio, sr=None, mono=True)
        return audio.astype(np.float32), sr


# =========================================================
# Feature extraction for DL emotion models (188 features)
# =========================================================
def extract_emotion_features(audio, sr):
    audio = audio.astype(np.float64)

    min_len = int(0.5 * sr)
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)

    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_mean = np.mean(mel, axis=1)

    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    try:
        y_harmonic = librosa.effects.harmonic(audio)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
    except Exception:
        tonnetz_mean = np.zeros(6)

    features = np.concatenate(
        [mfcc_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean]
    ).astype(np.float32)

    # 🔥 MUST MATCH TRAINING SIZE
    if len(features) < 188:
        features = np.pad(features, (0, 188 - len(features)))
    else:
        features = features[:188]

    return features


# =========================================================
# CORRECT Deep Learning Emotion Prediction
# =========================================================
def predict_emotion(audio, sr):
    """
    Handles sigmoid-based binary emotion models correctly
    and feeds 16 features into the meta model.
    """
    emotion, _, _ = predict_emotion_with_confidence(audio, sr)
    return emotion


def predict_emotion_with_confidence(audio, sr):
    """
    Uses Ensemble Logic from 04_deep_ensemble_meta_model.py:
    1. Multi-class DNN predictions (8 probs)
    2. Binary Emotion Models (1 prob each -> 8 probs)
    Total Meta Features: 16
    """

    # 1. Feature extraction
    features = extract_emotion_features(audio, sr)
    features = np.expand_dims(features, axis=0)  # (1, 188)
    
    # ---------------------------------------------------------
    # PART A: Multi-class DNN Predictions
    # ---------------------------------------------------------
    if multi_model is None:
        print("ERROR: Multi-class model not loaded")
        return "error", 0.0

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

        # Handle Sigmoid vs Softmax
        if probs.shape[1] == 1:
            # sigmoid -> output is prob of being "True" (i.e. 'emotion')
            p = float(probs[0][0])
            binary_preds.append(p)
        else:
            # softmax -> [p_not, p_yes]
            p_yes = float(probs[0][1])
            binary_preds.append(p_yes)

    binary_preds = np.array(binary_preds).reshape(1, -1) # Shape (1, 8)
    
    # ---------------------------------------------------------
    # PART C: Construct Meta Features
    # ---------------------------------------------------------
    # Stack horizontally: [Multi_Probs (8) | Binary_Probs (8)] -> Total 16
    meta_inputs = np.hstack([multi_val, binary_preds])

    # ---------------------------------------------------------
    # PART D: Meta Model Prediction
    # ---------------------------------------------------------
    if meta_model_logreg is None:
        print("ERROR: Meta Logistic Regression model not loaded")
        return "error", 0.0
        
    final_pred_probs = meta_model_logreg.predict_proba(meta_inputs)
    final_class_idx = np.argmax(final_pred_probs)
    
    confidence = float(final_pred_probs[0][final_class_idx])
    
    # Map index back to emotion label using the same order
    predicted_emotion = target_emotions[final_class_idx]
    
    # Create probability dictionary for Radar Chart
    probabilities = {
        emotion: float(prob) 
        for emotion, prob in zip(target_emotions, final_pred_probs[0])
    }
    
    # 🔒 Confidence-based rejection
    if confidence < 0.35: # Slightly lower threshold for LogReg as it tends to be more conservative
        return "uncertain", confidence, probabilities

    return predicted_emotion, confidence, probabilities


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


@app.route("/predict_gender", methods=["POST"])
def predict_gender():
    audio, sr = load_wav_from_bytes(request.files["file"].read())

    f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:
        return jsonify({"gender": "unknown", "confidence": 0.0, "pitch": 0.0})

    pitch = np.median(f0)
    gender = "male" if pitch < 165 else "female"

    # Calculate confidence based on distance from threshold
    threshold = 165
    if gender == "male":
        # For male: lower pitch = higher confidence (max at 50Hz, min at 165Hz)
        confidence = max(0.5, min(0.95, 1.0 - (pitch - 50) / (threshold - 50)))
    else:
        # For female: higher pitch = higher confidence (min at 165Hz, max at 300Hz)
        confidence = max(0.5, min(0.95, (pitch - threshold) / (300 - threshold)))

    return jsonify(
        {
            "gender": gender,
            "confidence": round(confidence, 3),
            "pitch": round(float(pitch), 2),
        }
    )


@app.route("/predict_emotion", methods=["POST"])
def predict_emotion_api():
    audio, sr = load_wav_from_bytes(request.files["file"].read())

    if not emotion_models or meta_model_logreg is None:
        return jsonify({"error": "DL emotion models not loaded"}), 500

    # Get emotion prediction with confidence
    emotion, confidence, probabilities = predict_emotion_with_confidence(audio, sr)

    return jsonify(
        {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "probabilities": probabilities,
            "model": "DeepLearning (Ensemble)",
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


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    app.run(debug=True, port=5001)

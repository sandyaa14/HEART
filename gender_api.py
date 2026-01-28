from flask import Flask, request, jsonify
import librosa
import numpy as np
import io
import soundfile as sf

app = Flask(__name__)

# -------------------------------
# Pitch → Gender logic (GOOD)
# -------------------------------
def predict_gender_from_pitch(pitch):
    if pitch < 140:
        return "male", 0.9
    elif pitch > 140 and pitch < 185:
        return "male", round(0.7 + (165 - pitch)/100, 3)
    elif pitch > 185 and pitch < 200 :
        return "female", round(0.7 + (pitch - 185)/100, 3)
    elif pitch > 200:
        return "uncertain", 0.55
    else:
        return "female", 0.9


# -------------------------------
# Audio loader
# -------------------------------
def load_audio_from_bytes(audio_bytes):
    bio = io.BytesIO(audio_bytes)
    try:
        audio, sr = sf.read(bio)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    except Exception:
        bio.seek(0)
        audio, sr = librosa.load(bio, sr=None, mono=True)

    return audio.astype(np.float32), sr


# -------------------------------
# Detect gender API
# -------------------------------
@app.route("/predict_gender", methods=["POST"])
def predict_gender():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    audio_bytes = request.files["file"].read()

    audio, sr = load_audio_from_bytes(audio_bytes)

    # Pitch extraction
    f0, _, _ = librosa.pyin(
        audio,
        fmin=50,
        fmax=400,
        sr=sr
    )

    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:
        return jsonify({
            "gender": "uncertain",
            "confidence": 0.0,
            "pitch": 0.0
        })

    pitch = float(np.median(f0))

    gender, confidence = predict_gender_from_pitch(pitch)

    return jsonify({
        "gender": gender,
        "confidence": round(confidence, 2),
        "pitch": round(pitch, 2)
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)

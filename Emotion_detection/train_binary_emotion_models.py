from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import pickle
import random
import tensorflow as tf

# ----------------------------
# Reproducibility / Seeds
# ----------------------------
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ----------------------------
# Paths (LOCAL – change if needed)
# ----------------------------
SAVE_FOLDER = r"C:\Final Year\Emotion_detection"
MODEL_SAVE_FOLDER = os.path.join(SAVE_FOLDER, "Models")

os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)

print("Loading features from:", SAVE_FOLDER)
print("Saving models to:", MODEL_SAVE_FOLDER)

# ----------------------------
# Helper: Convert labels to binary (one-vs-all)
# ----------------------------
def prepare_binary_labels(labels, target_emotion):
    """
    Convert multi-class labels into binary labels for a specific emotion.
    target_emotion -> 1
    all other emotions -> 0
    """
    return np.array([1 if label == target_emotion else 0 for label in labels])

# ----------------------------
# Helper: Build binary classifier model
# ----------------------------
def build_binary_model(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_shape,)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output: probability of target emotion

    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ----------------------------
# Load train & validation data
# ----------------------------
train_pkl_path = os.path.join(SAVE_FOLDER, 'ravdess_train.pkl')
val_pkl_path = os.path.join(SAVE_FOLDER, 'ravdess_val.pkl')

if not os.path.exists(train_pkl_path) or not os.path.exists(val_pkl_path):
    raise FileNotFoundError(
        f"Could not find ravdess_train.pkl or ravdess_val.pkl in {SAVE_FOLDER}. "
        "Make sure you ran the feature extraction script first."
    )

with open(train_pkl_path, 'rb') as f:
    train_data = pickle.load(f)

with open(val_pkl_path, 'rb') as f:
    val_data = pickle.load(f)

X_train = np.array([item[0] for item in train_data])
y_train = np.array([item[1] for item in train_data])

X_val = np.array([item[0] for item in val_data])
y_val = np.array([item[1] for item in val_data])

print("Train features shape:", X_train.shape)
print("Val features shape:", X_val.shape)

# ----------------------------
# Emotions (channels)
# ----------------------------
target_emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'calm']
input_shape = X_train.shape[1]  # Number of feature dimensions

channel_models = {}

# ----------------------------
# Train / Load one model per emotion
# ----------------------------
for i, emotion in enumerate(target_emotions):
    print("=" * 60)
    print(f"Channel {i+1}/{len(target_emotions)}: {emotion}")
    model_path = os.path.join(MODEL_SAVE_FOLDER, f"{emotion}_model.h5")

    if os.path.exists(model_path):
        print(f"✅ Model for '{emotion}' already exists. Loading pre-trained model...")
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        channel_models[emotion] = model
        print(f"Model for '{emotion}' loaded successfully from {model_path}\n")
        continue

    print(f"🚀 Training new model for emotion: '{emotion}'")

    # Prepare binary labels: this emotion vs all others
    y_train_bin = prepare_binary_labels(y_train, emotion)
    y_val_bin = prepare_binary_labels(y_val, emotion)

    # Build model
    model = build_binary_model(input_shape)

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        )
    ]

    # Train model
    history = model.fit(
        X_train, y_train_bin,
        validation_data=(X_val, y_val_bin),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Save model in dictionary
    channel_models[emotion] = model
    print(f"💾 Model for '{emotion}' saved successfully at: {model_path}\n")

print("=" * 60)
print("✅ All emotion models are ready and saved in:", MODEL_SAVE_FOLDER)

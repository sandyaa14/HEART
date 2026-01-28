# 03_train_multiclass_dnn.py

# Import necessary libraries
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np
import os
import pickle
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ----------------------------
# Paths (LOCAL)
# ----------------------------
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(SCRIPT_DIR, "Emotion_detection")
MODEL_SAVE_FOLDER = os.path.join(SAVE_FOLDER, "Models")

os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)

print("Loading data from:", SAVE_FOLDER)
print("Models will be saved to:", MODEL_SAVE_FOLDER)

# ----------------------------
# Load the processed data
# ----------------------------
try:
    train_pkl_path = os.path.join(SAVE_FOLDER, 'ravdess_train.pkl')
    val_pkl_path = os.path.join(SAVE_FOLDER, 'ravdess_val.pkl')
    test_pkl_path = os.path.join(SAVE_FOLDER, 'ravdess_test.pkl')

    with open(train_pkl_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(val_pkl_path, 'rb') as f:
        val_data = pickle.load(f)

    with open(test_pkl_path, 'rb') as f:
        test_data = pickle.load(f)

    # Separate features and labels
    X_train = np.array([item[0] for item in train_data])
    y_train_labels = np.array([item[1] for item in train_data])

    X_val = np.array([item[0] for item in val_data])
    y_val_labels = np.array([item[1] for item in val_data])

    X_test = np.array([item[0] for item in test_data])
    y_test_labels = np.array([item[1] for item in test_data])

    print("[OK] Data loaded successfully.")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train_labels shape: {y_train_labels.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val_labels shape: {y_val_labels.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test_labels shape: {y_test_labels.shape}")

    # ----------------------------
    # Encode labels as integers and one-hot
    # ----------------------------
    target_emotions = ['neutral', 'happy', 'sad', 'angry',
                       'fearful', 'disgust', 'surprised', 'calm']
    label_to_index = {emotion: i for i, emotion in enumerate(target_emotions)}

    y_train = np.array([label_to_index[e] for e in y_train_labels])
    y_val = np.array([label_to_index[e] for e in y_val_labels])
    y_test = np.array([label_to_index[e] for e in y_test_labels])

    y_train_one_hot = to_categorical(y_train, num_classes=len(target_emotions))
    y_val_one_hot = to_categorical(y_val, num_classes=len(target_emotions))
    y_test_one_hot = to_categorical(y_test, num_classes=len(target_emotions))

    print("[OK] Labels encoded and one-hot encoded successfully.")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train_one_hot shape: {y_train_one_hot.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_val_one_hot shape: {y_val_one_hot.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_test_one_hot shape: {y_test_one_hot.shape}")

    # ----------------------------
    # Define Multi-class DNN model
    # ----------------------------
    def build_multi_dnn(input_shape, num_classes):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(input_shape,)))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(num_classes, activation='softmax'))  # Multi-class output

        model.compile(
            optimizer=optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # Build multi-class DNN model
    input_shape = X_train.shape[1]
    num_classes = len(target_emotions)

    multi_model = build_multi_dnn(input_shape, num_classes)

    model_path = os.path.join(MODEL_SAVE_FOLDER, "multi_dnn_model.h5")

    # Define callbacks with increased patience
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        )
    ]

    # Train the multi-class model
    print("[TRAINING] Training multi-class DNN model...")
    history = multi_model.fit(
        X_train, y_train_one_hot,
        validation_data=(X_val, y_val_one_hot),
        epochs=200,     # You can reduce this if training is slow
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    test_loss, multi_dnn_test_acc = multi_model.evaluate(X_test, y_test_one_hot, verbose=0)
    print(f"\n[RESULT] Multi-class DNN Test Accuracy: {multi_dnn_test_acc * 100:.2f}%")

    # Generate and print classification report
    y_pred_probs = multi_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\n[REPORT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_emotions))
    print(f"[SAVED] Best model saved at: {model_path}")

except FileNotFoundError:
    print(f"[ERROR] Data files not found in {SAVE_FOLDER}. "
          f"Make sure ravdess_train.pkl, ravdess_val.pkl, and ravdess_test.pkl exist.")
except Exception as e:
    print(f"[WARNING] An error occurred during Multi-class DNN training: {e}")

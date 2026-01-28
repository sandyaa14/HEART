import pickle
import numpy as np

# Load features (list of (feature_vector, emotion_label))
data = pickle.load(open("Emotion_detection/ravdess_train.pkl", "rb"))

# Extract ONLY feature vectors
features = np.array([item[0] for item in data])

# Load generated filenames
filenames = [line.strip() for line in open("filenames_sorted.txt", "r")]

# Gender rule: EVEN actor → FEMALE, ODD actor → MALE
def gender_from_filename(name):
    actor_id = int(name.split("-")[6].replace(".wav", ""))
    return 0 if actor_id % 2 == 0 else 1   # 0=female, 1=male

gender_labels = np.array([gender_from_filename(name) for name in filenames])

# Split & train
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    features, gender_labels, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("\n===================================")
print(f"Gender Classification Accuracy: {acc*100:.2f}%")
print("===================================\n")

pickle.dump(model, open("gender_model.pkl", "wb"))
print("Saved gender_model.pkl successfully!")

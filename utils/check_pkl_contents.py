import joblib

p = joblib.load("Emotion_detection/ravdess_train.pkl")
print(type(p))

if isinstance(p, dict):
    print("Keys:", p.keys())
else:
    print("Value:", p)

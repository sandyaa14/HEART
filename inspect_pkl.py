import pickle

data = pickle.load(open("Emotion_detection/ravdess_train.pkl", "rb"))

print("Total samples in ravdess_train.pkl:", len(data))

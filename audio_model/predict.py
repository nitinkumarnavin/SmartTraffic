import librosa
import numpy as np
import joblib

model = joblib.load(
    "audio_model/siren_model.pkl"
)

file_path = input(
    "Enter audio file path: "
)

audio, sr = librosa.load(
    file_path,
    sr=22050
)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

feature = np.mean(
    mfcc.T,
    axis=0
)

feature = feature.reshape(1, -1)

prediction = model.predict(
    feature
)[0]

labels = {
    0: "Ambulance",
    1: "Police",
    2: "Traffic Noise"
}

print(
    "\nPrediction:",
    labels[prediction]
)
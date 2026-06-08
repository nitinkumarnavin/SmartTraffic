import joblib
import librosa
import numpy as np

def detect_siren(audio_path):

    model = joblib.load("audio_model/siren_model.pkl")

    audio, sr = librosa.load(audio_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    feature = np.mean(mfcc.T, axis=0)

    prediction = model.predict([feature])[0]

    labels = {
        0: "Ambulance",
        1: "Police",
        2: "Firetruck",
        3: "Traffic Noise",
        4: "Background"
    }

    return labels[prediction]
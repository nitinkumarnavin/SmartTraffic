import sounddevice as sd
import numpy as np
import librosa
import joblib

model = joblib.load(
    "audio_model/siren_model.pkl"
)

labels = {
    0: "Ambulance",
    1: "Police",
    2: "Firetruck",
    3: "Traffic Noise",
    4: "Background"
}

duration = 5
sample_rate = 22050

print("Listening...")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype='float32'
)

sd.wait()

audio = audio.flatten()

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
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

print(
    "\nDetected:",
    labels[prediction]
)
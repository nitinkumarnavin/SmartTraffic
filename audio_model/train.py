import os
import librosa
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_PATH = "dataset"

X = []
y = []

labels = {
    "ambulance": 0,
    "police": 1,
    "traffic_noise": 2
}

for folder_name, label in labels.items():

    folder_path = os.path.join(DATASET_PATH, folder_name)

    for file in os.listdir(folder_path):

        if file.endswith(".mp3") or file.endswith(".wav"):

            file_path = os.path.join(folder_path, file)

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

            X.append(feature)
            y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy:", accuracy)

joblib.dump(
    model,
    "audio_model/siren_model.pkl"
)

print("Model Saved Successfully")
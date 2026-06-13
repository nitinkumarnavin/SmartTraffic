import os
from config import DATASET_PATH

for seq in os.listdir(DATASET_PATH):
    seq_path = os.path.join(DATASET_PATH, seq)

    if os.path.isdir(seq_path):
        images = os.listdir(seq_path)

        if len(images) == 0:
            print(f"[ERROR] {seq} is empty")
        else:
            print(f"[OK] {seq}: {len(images)} images")
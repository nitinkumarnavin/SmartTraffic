import os
from config import DATASET_PATH

total_images = 0
sequences = os.listdir(DATASET_PATH)

print(f"Total Sequences: {len(sequences)}\n")

for seq in sequences:
    seq_path = os.path.join(DATASET_PATH, seq)

    if os.path.isdir(seq_path):
        count = len(os.listdir(seq_path))
        total_images += count
        print(f"{seq}: {count} images")

print(f"\nTotal Images: {total_images}")
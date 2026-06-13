import os
from config import DATASET_PATH

counts = []

for seq in os.listdir(DATASET_PATH):
    seq_path = os.path.join(DATASET_PATH, seq)

    if os.path.isdir(seq_path):
        counts.append(len(os.listdir(seq_path)))

print("Dataset Statistics")
print("------------------")
print("Sequences:", len(counts))
print("Min Images:", min(counts))
print("Max Images:", max(counts))
print("Average Images:", sum(counts) / len(counts))
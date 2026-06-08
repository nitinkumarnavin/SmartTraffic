import os

dataset_path = r"C:\Users\KOLA JATIN\.cache\kagglehub\datasets\bratjay\ua-detrac-orig\versions\2\DETRAC-Images"

folders = os.listdir(dataset_path)

print("Total folders:", len(folders))

for folder in folders[:10]:
    print(folder)
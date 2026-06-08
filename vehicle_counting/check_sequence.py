import os

path = r"C:\Users\KOLA JATIN\.cache\kagglehub\datasets\bratjay\ua-detrac-orig\versions\2\DETRAC-Images\DETRAC-Images\MVI_20011"

files = os.listdir(path)

print("Total images:", len(files))
print("First 10 images:")

for f in files[:10]:
    print(f)
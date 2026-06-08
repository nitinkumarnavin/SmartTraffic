import os

path = r"C:\Users\KOLA JATIN\.cache\kagglehub\datasets\bratjay\ua-detrac-orig\versions\2\DETRAC-Images"

for item in os.listdir(path):
    print(item)
    
    full_path = os.path.join(path, item)
    
    if os.path.isdir(full_path):
        print("\nContents:")
        for sub in os.listdir(full_path)[:20]:
            print("  ", sub)
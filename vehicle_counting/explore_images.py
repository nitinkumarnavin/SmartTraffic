import os
import cv2
from config import DATASET_PATH

sequence = os.listdir(DATASET_PATH)[0]
seq_path = os.path.join(DATASET_PATH, sequence)

image_name = os.listdir(seq_path)[0]
image_path = os.path.join(seq_path, image_name)

img = cv2.imread(image_path)

cv2.imshow("Sample Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
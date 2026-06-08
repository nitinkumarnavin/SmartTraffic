import cv2
from ultralytics import YOLO

print("OpenCV OK")
print("YOLO OK")

model = YOLO("yolov8n.pt")

print("Model Loaded Successfully")
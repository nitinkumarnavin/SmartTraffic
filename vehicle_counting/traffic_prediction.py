from ultralytics import YOLO
import cv2
import os

# -------------------------
# MODEL
# -------------------------
model = YOLO("yolov8n.pt")

# -------------------------
# VIDEO PATH
# -------------------------
video_path = r"C:\personal inside\projects\SmartTraffic\vehicle_counting\test(1).mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video not found")
    exit()

# -------------------------
# SETTINGS
# -------------------------
vehicle_classes = ["car", "truck", "bus", "motorcycle"]
class_names = model.names

history = []   # stores vehicle counts

# -------------------------
# LOOP
# -------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.25, imgsz=640, verbose=False)

    vehicle_count = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]

            if class_name in vehicle_classes:
                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # -------------------------
    # STORE HISTORY
    # -------------------------
    history.append(vehicle_count)

    if len(history) > 10:
        history.pop(0)

    # -------------------------
    # SMOOTHING (noise reduction)
    # -------------------------
    avg_count = sum(history) / len(history)

    # -------------------------
    # PREDICTION LOGIC
    # -------------------------
    if avg_count < 10:
        prediction = "LOW TRAFFIC"
        color = (0, 255, 0)

    elif avg_count < 25:
        prediction = "MEDIUM TRAFFIC"
        color = (0, 255, 255)

    else:
        prediction = "HIGH TRAFFIC"
        color = (0, 0, 255)

    # -------------------------
    # DISPLAY INFO
    # -------------------------
    cv2.putText(frame,
                f"Vehicles: {vehicle_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                f"Avg Traffic: {avg_count:.1f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2)

    cv2.putText(frame,
                f"Prediction: {prediction}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3)

    cv2.imshow("Traffic Prediction Model", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# -------------------------
# LOAD MODEL
# -------------------------
model = YOLO("yolov8n.pt")

# -------------------------
# VIDEO PATH
# -------------------------
video_path = r"C:\personal inside\projects\SmartTraffic\vehicle_counting\test(1).mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# -------------------------
# SETTINGS
# -------------------------
vehicle_classes = ["car", "truck", "bus", "motorcycle"]
class_names = model.names

frame_skip = 2
frame_count = 0

# -------------------------
# DATA STORAGE FOR GRAPH
# -------------------------
vehicle_history = []
density_history = []

# -------------------------
# PROCESS VIDEO
# -------------------------
while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame_count += 1

    # Skip frames for faster processing
    if frame_count % frame_skip != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (960, 540))

    # YOLO Detection
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

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    # -------------------------
    # TRAFFIC DENSITY LOGIC
    # -------------------------
    density_percentage = min((vehicle_count / 50) * 100, 100)

    if density_percentage < 30:
        density = "LOW"
        color = (0, 255, 0)

    elif density_percentage < 70:
        density = "MEDIUM"
        color = (0, 255, 255)

    else:
        density = "HIGH"
        color = (0, 0, 255)

    # Save data for graph
    vehicle_history.append(vehicle_count)
    density_history.append(density_percentage)

    # -------------------------
    # DISPLAY RESULTS
    # -------------------------
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Density: {density}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.putText(
        frame,
        f"Density %: {density_percentage:.1f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Traffic Density Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------
# CLEANUP
# -------------------------
cap.release()
cv2.destroyAllWindows()

# -------------------------
# DENSITY GRAPH
# -------------------------
plt.figure(figsize=(10, 5))

plt.plot(density_history)

plt.title("Traffic Density Percentage Over Time")
plt.xlabel("Frame Number")
plt.ylabel("Density Percentage (%)")

plt.grid(True)

plt.savefig("traffic_density_graph.png")

plt.show()

print("Traffic density graph saved as traffic_density_graph.png")
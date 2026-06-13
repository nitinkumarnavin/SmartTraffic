from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Video path
video_path = r"vehicle_counting/test(1).mp4"

cap = cv2.VideoCapture(video_path)

vehicle_classes = ["car", "truck", "bus", "motorcycle"]

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # Resize for better detection of small vehicles
    #frame = cv2.resize(frame, (1280, 720))

    # Vehicle tracking
    results = model(
        frame,
        conf=0.15,
        iou=0.45,
        imgsz=1280,
        verbose=False
    )

    vehicle_count = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name in vehicle_classes:
                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    # Vehicle count display
    cv2.putText(
        frame,
        f"Vehicles Detected: {vehicle_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    cv2.imshow("Traffic Vehicle Detection", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
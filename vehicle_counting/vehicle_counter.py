from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8m.pt")

video_path = r"dataset/traffic_videos/Traffic Intersection Drone View - 4K [bfc2wsX29zk].mp4"

cap = cv2.VideoCapture(video_path)

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    car_count = 0
    bike_count = 0
    bus_count = 0
    truck_count = 0

    for box in results[0].boxes:

        cls = int(box.cls[0])

        if cls == 2:
            car_count += 1
        elif cls == 3:
            bike_count += 1
        elif cls == 5:
            bus_count += 1
        elif cls == 7:
            truck_count += 1

    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        f"Cars: {car_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Bikes: {bike_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Buses: {bus_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Trucks: {truck_count}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Vehicle Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
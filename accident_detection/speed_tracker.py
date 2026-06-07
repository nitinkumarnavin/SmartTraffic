from ultralytics import YOLO
import cv2
import math

model = YOLO("yolov8n.pt")

video_path = "accident_detection/videos/test.mp4"

cap = cv2.VideoCapture(video_path)

previous_positions = {}
slow_counter = {}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True
    )

    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if track_id in previous_positions:

                old_x, old_y = previous_positions[track_id]

                distance = math.sqrt(
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                )

                cv2.putText(
                    annotated_frame,
                    f"Speed:{distance:.1f}",
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # Accident Alert Logic
                if distance < 2:

                    slow_counter[track_id] = slow_counter.get(track_id, 0) + 1

                else:

                    slow_counter[track_id] = 0

                if slow_counter[track_id] > 20:

                    cv2.putText(
                        annotated_frame,
                        "ACCIDENT ALERT!",
                        (center_x, center_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

            previous_positions[track_id] = (
                center_x,
                center_y
            )

    cv2.imshow(
        "Speed Tracking",
        annotated_frame
    )

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
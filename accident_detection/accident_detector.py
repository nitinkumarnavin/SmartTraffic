from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video_path = "accident_detection/videos/test.mp4"

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True
    )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Vehicle Tracking",
        annotated_frame
    )

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
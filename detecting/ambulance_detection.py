from ultralytics import YOLO
import cv2

# Load trained ambulance model
model = YOLO(r"C:\Users\KOLA JATIN\runs\detect\train-3\weights\best.pt")

# Video path
video_path = r"C:\personal inside\projects\SmartTraffic\detecting\t[1].webm"

cap = cv2.VideoCapture(video_path)

# Get original FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer (9:16 portrait)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "ambulance_output.mp4",
    fourcc,
    fps,
    (540, 960)
)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Resize to portrait size (9:16)
    frame = cv2.resize(frame, (540, 960))

    # Detect ambulance
    results = model.predict(
        frame,
        conf=0.25,      # Lower confidence for better recall
        imgsz=640,
        verbose=False
    )

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Save output video
    out.write(annotated_frame)

    # Show live detection
    cv2.imshow("Ambulance Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Detection completed!")
print("Output saved as: ambulance_output.mp4")
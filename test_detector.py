import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Open a test video file or use 0 for your webcam
# (Replace 'test_traffic.mp4' with an actual path to a video file on your computer)
cap = cv2.VideoCapture("test_traffic.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video ended or cannot be opened.")
        break

    # Run inference: COCO classes -> 2: car, 3: motorcycle, 5: bus, 7: truck
    results = model(frame, classes=[2, 3, 5, 7])

    # Visualize the detection bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the live window
    cv2.imshow("Traffic Analyzer", annotated_frame)

    # Press 'q' on your keyboard to exit the window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
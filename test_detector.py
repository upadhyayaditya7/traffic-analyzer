import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "test_traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Explicit check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'. Please check if it exists in the folder.")
    exit()

print("Video loaded successfully. Processing frames... Press 'q' in the video window to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video ended or cannot be read further.")
        break

    # Run inference: COCO classes -> 2: car, 3: motorcycle, 5: bus, 7: truck
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)

    # Visualize the detection bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the live window
    cv2.imshow("Traffic Analyzer", annotated_frame)

    # Press 'q' on your keyboard to exit the window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting by user request.")
        break

cap.release()
cv2.destroyAllWindows()
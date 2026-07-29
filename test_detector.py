import cv2
from ultralytics import solutions

# Open the video file
video_path = "test_traffic.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(
        f"Error: Could not open video file '{video_path}'. Please check if it exists in the folder."
    )
    exit()

print(
    "Video loaded successfully. Processing frames... Press 'q' in the video window to exit."
)

# Define a counting line coordinates [x1, y1, x2, y2] across your road view
line_points = [(20, 400), (1200, 400)]

# Initialize the Ultralytics Object Counter with the model and classes built right in
counter = solutions.ObjectCounter(
    show=False,
    region=line_points,
    classes=[2, 3, 5, 7],  # Restrict to car, motorcycle, bus, truck
    model="yolov8n.pt",
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video ended or cannot be read further.")
        break

    # The counter handles detection, tracking, and line-crossing automatically
    results = counter(frame)
    annotated_frame = results.plot_im

    # Display the live window
    cv2.imshow("Traffic Analyzer - Vehicle Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting by user request.")
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import solutions

# Open the video file
video_path = "test_drone_view.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(
        f"Error: Could not open video file '{video_path}'. Please check if it exists in the folder."
    )
    exit()

# Dynamically get the actual width and height of this specific video file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Loaded Successfully. Resolution: {frame_width}x{frame_height}")

# Automatically place the counting line horizontally across the lower-middle part of the screen
line_points = [
    (int(frame_width * 0.1), int(frame_height * 0.75)),
    (int(frame_width * 0.9), int(frame_height * 0.75)),
]

# Initialize the Ultralytics Object Counter
counter = solutions.ObjectCounter(
    show=False,
    region=line_points,
    classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
    model="yolov8n.pt",
)

print("Processing frames... Press 'q' in the video window to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video ended or cannot be read further.")
        break

    results = counter(frame)
    annotated_frame = results.plot_im

    # Resize the high-res frame down so it fits completely inside your display window
    resized_frame = cv2.resize(annotated_frame, (1280, 720))

    # Display the resized live window
    cv2.imshow("Traffic Analyzer - Vehicle Counter", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting by user request.")
        break

cap.release()
cv2.destroyAllWindows()
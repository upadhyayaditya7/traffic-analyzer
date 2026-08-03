import cv2
from ultralytics import solutions

# Open the video file
video_path = "test_side_view.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(
        f"Error: Could not open video file '{video_path}'. Please check if it exists in the folder."
    )
    exit()

# Get original dimensions
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original Resolution: {orig_width}x{orig_height}")

PROCESSING_WIDTH = 960
PROCESSING_HEIGHT = 540

# Setting region points completely outside the frame view so the line is invisible
line_points = [(-10, -10), (-20, -20)]

# Initialize the Ultralytics Object Counter with hidden line points
counter = solutions.ObjectCounter(
    show=False,
    region=line_points,
    classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
    model="yolov8n.pt",
)

print("Processing frames... Press 'q' in the video window to exit.")

frame_count = 0
skip_frames = 3

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video ended or cannot be read further.")
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    small_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))

    results = counter(small_frame)
    annotated_frame = results.plot_im

    cv2.imshow("Traffic Analyzer - Vehicle Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting by user request.")
        break

cap.release()
cv2.destroyAllWindows()
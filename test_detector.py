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

orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original Resolution: {orig_width}x{orig_height}")

PROCESSING_WIDTH = 960
PROCESSING_HEIGHT = 540

# Place the counting line across the middle where vehicles cross it
line_points = [
    (int(PROCESSING_WIDTH * 0.1), int(PROCESSING_HEIGHT * 0.48)),
    (int(PROCESSING_WIDTH * 0.9), int(PROCESSING_HEIGHT * 0.48)),
]

# Initialize the Ultralytics Object Counter
counter = solutions.ObjectCounter(
    show=False,
    region=line_points,
    classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
    model="yolov8n.pt",
)

# Create a resizable OpenCV window so it can scale to full screen cleanly
window_name = "Traffic Analyzer - Vehicle Counter"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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

    # Resize down for fast AI processing
    small_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))

    results = counter(small_frame)
    annotated_frame = results.plot_im

    # Safely retrieve individual class counts from the counter object's dictionary
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0

    if hasattr(counter, "class_wise_count") and isinstance(counter.class_wise_count, dict):
        counts_dict = counter.class_wise_count
        car_count = counts_dict.get("car", counts_dict.get(2, 0))
        motorcycle_count = counts_dict.get("motorcycle", counts_dict.get(3, 0))
        bus_count = counts_dict.get("bus", counts_dict.get(5, 0))
        truck_count = counts_dict.get("truck", counts_dict.get(7, 0))

    total_tracked = getattr(counter, "in_count", 0) + getattr(counter, "out_count", 0)

    # Draw our clean custom dashboard box on top
    cv2.rectangle(annotated_frame, (20, 20), (320, 160), (0, 0, 0), -1)
    cv2.rectangle(annotated_frame, (20, 20), (320, 160), (0, 255, 0), 2)

    cv2.putText(
        annotated_frame,
        "--- LIVE COUNTS ---",
        (35, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Total Tracked: {total_tracked}",
        (35, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Cars: {car_count}",
        (35, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"2-Wheelers: {motorcycle_count}",
        (35, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Trucks/Buses: {truck_count + bus_count}",
        (35, 153),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Resize back up to match the original video frame size so it displays crisp at full size
    display_frame = cv2.resize(annotated_frame, (orig_width, orig_height))

    cv2.imshow(window_name, display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting by user request.")
        break

cap.release()
cv2.destroyAllWindows()
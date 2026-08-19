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

    # Extract individual class counts from the counter object securely
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0

    if hasattr(counter, "class_wise_count") and counter.class_wise_count:
        car_count = counter.class_wise_count.get("car", 0)
        motorcycle_count = counter.class_wise_count.get("motorcycle", 0)
        bus_count = counter.class_wise_count.get("bus", 0)
        truck_count = counter.class_wise_count.get("truck", 0)

    # Draw a clean dark background box with a green border for the on-screen dashboard overlay
    cv2.rectangle(annotated_frame, (20, 20), (320, 160), (0, 0, 0), -1)
    cv2.rectangle(annotated_frame, (20, 20), (320, 160), (0, 255, 0), 2)

    # Render live vehicle categories and totals onto the annotated video frame
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
        f"Cars: {car_count}",
        (35, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"2-Wheelers: {motorcycle_count}",
        (35, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Buses: {bus_count}",
        (35, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Trucks: {truck_count}",
        (35, 153),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Traffic Analyzer - Vehicle Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting by user request.")
        break

cap.release()
cv2.destroyAllWindows()
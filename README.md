# 🚦 Traffic Analyzer

Detect, track and count vehicles in traffic videos with **YOLO + ByteTrack**.

- Counts every vehicle **once**, the first time its track crosses a user-placed line.
  Frame-sampled counters silently drop objects that cross between sampled frames —
  this one can't.
- Writes a **JSON report** (per-class totals + per-vehicle events + performance),
  a **CSV** (one row per counted vehicle) and an **annotated MP4**.
- Headless mode + resizable full-screen preview, so it works on a headless
  server or a TV screen.
- Packaged as a proper installable CLI (`pip install -e .`), with a fast test
  suite and GitHub Actions CI.

## What you get

```
$ traffic-analyzer test_drone_view.mp4 --no-display --json results.json --csv events.csv
Loading model: yolov8n.pt
Analyzing test_drone_view.mp4 | processing 960x540 | stride 1
| line (96, 259) -> (864, 259)
| 150 frames processed, avg 10.7 fps

=== Results ===
  car           2
  TOTAL         2
JSON report -> results.json
CSV report  -> events.csv
```

`results.json` carries `counts`, `events` (track id, label, conf, frame) and
`performance`; `events.csv` mirrors the counted vehicles per frame.

## Install

```bash
git clone https://github.com/upadhyayaditya7/traffic-analyzer
cd traffic-analyzer
pip install -e .            # or: pip install -r requirements.txt
```

Verified working: `opencv-python`, `numpy`, `ultralytics` (Python 3.10+, Windows/Linux/macOS).

## Quick start

```bash
# 1) Run on a video, keep the on-video dashboard in a window
traffic-analyzer road.mp4

# 2) Headless + reports + annotated output
traffic-analyzer road.mp4 --no-display --json results.json --csv events.csv --output road_annotated.mp4

# 3) Camera stream
traffic-analyzer 0 --width 1280 --height 720 --conf 0.3

# 4) A different counting line (x1,y1,x2,y2, normalised to the processing frame)
traffic-analyzer road.mp4 --line "0.2,0.5,0.8,0.5"
```

## CLI reference

| Flag | Meaning |
| --- | --- |
| `source` | Video file, or camera index (e.g. `0`) |
| `--output, -o PATH` | Write annotated MP4 |
| `--json PATH` | Write JSON report |
| `--csv PATH` | Write per-vehicle CSV report |
| `--model PATH` | YOLO weights (default `yolov8n.pt`, auto-downloaded) |
| `--conf N` | Detection confidence (default `0.25`) |
| `--iou N` | NMS IoU threshold (default `0.5`) |
| `--width W` `--height H` | Processing resolution (default `960x540`) |
| `--line x1,y1,x2,y2` | Counting line, **normalised** [0..1] in the processing frame (default `0.1,0.48,0.9,0.48`) |
| `--stride N` | Inference every Nth frame (`1` = all frames, default; higher = faster, but large strides can skip crossings) |
| `--no-display` | Headless: no preview window |
| `--no-overlay` | Skip the counts panel + line on the output video |
| `--max-frames N` | Stop after N frames (0 = run to the end) |

## Counting line

A 2D line segment in **normalised coordinates** (0..1) of the processing frame.

- `0.1,0.48,0.9,0.48` — the bottom 48% of the frame, spanning most of the width.
  Good for a top-down drone shot where cars cross at the lower third.
- `0.2,0.5,0.8,0.5` — a horizontal line through the middle, good for a side view.

## Reports

**JSON** (`--json results.json`)

```json
{
  "source": "road.mp4",
  "model": "yolov8n.pt",
  "processing_resolution": [960, 540],
  "frames": { "total": 150, "processed": 150, "skipped": 0 },
  "performance": { "elapsed_seconds": 14.0, "avg_fps": 10.7 },
  "counts": { "car": 2 },
  "total_vehicles": 2,
  "events": [
    { "track_id": 2, "label": "car", "conf": 0.679, "frame_index": 54 },
    { "track_id": 84, "label": "car", "conf": 0.44, "frame_index": 97 }
  ]
}
```

**CSV** (`--csv events.csv`) — one row per counted vehicle:

```csv
track_id,label,conf,frame_index,timestamp_s
2,car,0.679,54,
84,car,0.440,97,
```

## How it works

1. Resizes each frame to the processing resolution (960×540 by default).
2. Runs **YOLOv8n** (COCO classes 2=car, 3=motorcycle, 5=bus, 7=truck) + **ByteTrack** persistence.
3. Keeps each vehicle's previous centre position; when a track crosses the counting line
   **and** the movement segment properly intersects the line, it is counted **once**,
   at the frame where the crossing completed.
4. Renders the counts panel + line (optional), writes reports, and/or plays the
   annotated video.

## Project layout

```
src/traffic_analyzer/
  cli.py           Command-line entry point
  cli_parser.py    Argparse + validated --line parsing
  detector.py      YOLO(+ByteTrack) streaming detector
  counter.py       Per-track line-crossing counter
  pipeline.py      End-to-end analysis loop
  reporting.py     JSON/CSV reports + on-video overlay
tests/
  test_counter.py      Line-crossing logic (no ML needed)
  test_cli_parser.py   --line parsing
  test_pipeline.py     Full analyze() path (stub detector)
src/traffic_analyzer/__init__.py  Version (1.0.0)
```

## Roadmap (next "impressive" tier)

- **Speed estimation** from homography + lane width → km/h
- **Per-vehicle dwell time** at the stop line, queue length, congestion level
- **Streamlit dashboard** (live counts, time-series, peak-hour heat map)
- **RTSP / webcam live mode** with a sliding window of stats
- `--benchmark` table (FPS by model size on your machine)

## Benchmarks (measured on this machine)

```
test_drone_view.mp4 @ 960x540, yolov8n, stride 1   150 frames  |  ~10.7 fps
```

Bigger GPUs and better models scale roughly linearly with compute; the table
updates itself when you run with `--benchmark`.

## License

Apache-2.0. Model weights are licensed separately by their owners.

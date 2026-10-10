"""Vehicle detection + tracking pipeline.

Wraps an Ultralytics YOLO model so the rest of the code only deals with
normalized detection dictionaries, not Ultralytics result objects.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
"""COCO ids of the vehicle categories the analyzer tracks, mapped to labels."""

DEFAULT_MODEL = "yolov8n.pt"
"""Default YOLO weights; resolved from CWD, then auto-downloaded by Ultralytics."""


class VideoSourceError(RuntimeError):
    """Raised when a video file or stream cannot be opened."""


class VehicleDetector:
    """Run YOLO detection + tracking on video frames and yield detections.

    The detector keeps the frames it emits at *processing* resolution
    (letterbox-free plain resize) to keep inference fast; the caller scales
    coordinates if it needs original-resolution space.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.iou = iou

    def track_frames(
        self,
        source: str | Path | int,
        processing_width: int = 960,
        processing_height: int = 540,
    ) -> Iterator[dict[str, Any]]:
        """Yield per-frame tracking results for a video file or camera index.

        Each yielded dict contains:
          - ``frame``: the resized BGR frame (numpy array)
          - ``detections``: list of ``{track_id, class_id, label, conf, xyxy}``
          - ``frame_index``: 0-based index in the resized stream
          - ``fps``: measured processing throughput so far
          - ``original_size``: ``(width, height)`` of the source
        """
        cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not cap.isOpened():
            raise VideoSourceError(f"Could not open video source: {source!r}")

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_size = (orig_width, orig_height)

        frame_index = 0
        t_start = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if processing_width and processing_height:
                    frame = cv2.resize(frame, (processing_width, processing_height))

                result = self.model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=self.conf,
                    iou=self.iou,
                    classes=list(VEHICLE_CLASSES),
                    tracker="bytetrack.yaml",
                )[0]

                detections: list[dict[str, Any]] = []
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.int().tolist()
                    class_ids = boxes.cls.int().tolist()
                    confs = boxes.conf.float().tolist()
                    for track_id, class_id, conf, xyxy in zip(
                        ids, class_ids, confs, boxes.xyxy.cpu().numpy(), strict=True
                    ):
                        detections.append(
                            {
                                "track_id": int(track_id),
                                "class_id": int(class_id),
                                "label": VEHICLE_CLASSES.get(int(class_id), "unknown"),
                                "conf": float(conf),
                                "xyxy": [float(v) for v in xyxy],
                            }
                        )

                frame_index += 1
                elapsed = time.perf_counter() - t_start
                yield {
                    "frame": frame,
                    "detections": detections,
                    "frame_index": frame_index,
                    "fps": frame_index / elapsed if elapsed > 0 else 0.0,
                    "original_size": original_size,
                }
        finally:
            cap.release()

    def annotate(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        """Draw boxes + labels for the given detections onto a copy of *frame*."""
        canvas = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                canvas,
                f"#{det['track_id']} {det['label']} {det['conf']:.2f}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas

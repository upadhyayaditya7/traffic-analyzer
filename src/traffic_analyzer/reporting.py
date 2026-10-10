"""Result sink + overlay drawing.

Collects per-frame crossing events, writes JSON/CSV reports, and renders an
optional on-video dashboard (counts panel + counting line).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2


@dataclass
class CrossEvent:
    """One vehicle counted at a specific frame."""

    track_id: int
    label: str
    conf: float
    frame_index: int
    fps_at_event: float = 0.0
    source_timestamp: float | None = None


@dataclass
class Results:
    """Aggregated analysis results, serializable to JSON/CSV."""

    source: str
    model: str
    processed_width: int
    processed_height: int
    total_frames: int = 0
    processed_frames: int = 0
    skipped_frames: int = 0
    elapsed_seconds: float = 0.0
    avg_fps: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    events: list[CrossEvent] = field(default_factory=list)

    @property
    def total_vehicles(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "model": self.model,
            "processing_resolution": [self.processed_width, self.processed_height],
            "frames": {
                "total": self.total_frames,
                "processed": self.processed_frames,
                "skipped": self.skipped_frames,
            },
            "performance": {
                "elapsed_seconds": round(self.elapsed_seconds, 2),
                "avg_fps": round(self.avg_fps, 2),
            },
            "counts": dict(sorted(self.counts.items(), key=lambda kv: -kv[1])),
            "total_vehicles": self.total_vehicles,
            "events": [
                {
                    "track_id": e.track_id,
                    "label": e.label,
                    "conf": round(e.conf, 3),
                    "frame_index": e.frame_index,
                    "source_timestamp": round(e.source_timestamp, 2)
                    if e.source_timestamp is not None
                    else None,
                }
                for e in self.events
            ],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["track_id", "label", "conf", "frame_index", "timestamp_s"])
            for e in self.events:
                ts = (
                    f"{e.source_timestamp:.2f}"
                    if e.source_timestamp is not None
                    else ""
                )
                writer.writerow([e.track_id, e.label, f"{e.conf:.3f}", e.frame_index, ts])


class Overlay:
    """Draws the counting line + counts panel onto annotated frames."""

    LINE_COLOR = (0, 215, 255)  # amber-ish BGR
    PANEL_BG = (30, 30, 30)
    PANEL_BORDER = (80, 220, 80)
    TEXT_MAIN = (255, 255, 255)
    TEXT_HEAD = (0, 215, 255)

    def __init__(self, line: tuple[tuple[int, int], tuple[int, int]]) -> None:
        self.line = line

    def draw(self, frame, detections, counts: dict[str, int], total: int, fps: float):
        canvas = frame.copy()
        (ax, ay), (bx, by) = self.line
        cv2.line(canvas, (ax, ay), (bx, by), self.LINE_COLOR, 2, cv2.LINE_AA)

        # Counts panel (top-left).
        x0, y0, w = 16, 16, 250
        pad, row_h = 10, 22
        entries = sorted(counts.items(), key=lambda kv: -kv[1])
        h = pad * 2 + 24 + 24 + len(entries) * row_h + 6
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), self.PANEL_BG, -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), self.PANEL_BORDER, 1)
        ty = y0 + pad + 16
        cv2.putText(canvas, "TRAFFIC ANALYZER", (x0 + pad, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.TEXT_HEAD, 1, cv2.LINE_AA)
        ty += 24
        cv2.putText(canvas, f"Total: {total}   FPS: {fps:.1f}", (x0 + pad, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.TEXT_MAIN, 1, cv2.LINE_AA)
        for label, n in entries:
            ty += row_h
            cv2.putText(canvas, f"{label}: {n}", (x0 + pad + 6, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def draw_detections(frame, detections):
        canvas = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 200, 90), 1)
            cv2.putText(canvas, f"#{det['track_id']} {det['label']}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 200, 90), 1, cv2.LINE_AA)
        return canvas

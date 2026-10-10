"""End-to-end pipeline test with a stub detector (no YOLO weights needed).

Exercises the full analyze() path: tracking chunks -> line-cross counting ->
annotated video + JSON + CSV outputs. YOLO itself is exercised by the manual
CLI run; CI runs the fast suite below on every push.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from traffic_analyzer.pipeline import analyze


class FakeDetector:
    """Yields synthetic tracking chunks: 3 vehicles crossing the line."""

    def __init__(self, width=960, height=540, frames=30):
        self.width = width
        self.height = height
        self.frames = frames
        self.model = "fake-model.pt"

    def track_frames(self, source, processing_width, processing_height):
        for idx in range(1, self.frames + 1):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            dets = []
            # Three vehicles moving downward across y=270 (the line row).
            for k, x in enumerate((200, 480, 760)):
                y = 40 + idx * 10 + k * 5
                dets.append(
                    {
                        "track_id": k + 1,
                        "class_id": 2,
                        "label": "car",
                        "conf": 0.9,
                        "xyxy": [x - 20, y - 20, x + 20, y + 20],
                    }
                )
            yield {
                "frame": frame,
                "detections": dets,
                "frame_index": idx,
                "fps": 30.0,
                "original_size": (self.width, self.height),
            }


@pytest.fixture()
def line():
    return ((96, 270), (864, 270))


def test_pipeline_counts_all_three_vehicles(tmp_path: Path, line):
    results = analyze(
        FakeDetector(),
        "synthetic",
        line,
        output_path=tmp_path / "out.mp4",
        display=False,
        progress_every=0,
    )
    assert results.total_vehicles == 3
    assert results.counts == {"car": 3}
    assert results.processed_frames == 30
    assert {e.track_id for e in results.events} == {1, 2, 3}


def test_pipeline_writes_json_and_csv(tmp_path: Path, line):
    results = analyze(
        FakeDetector(), "synthetic", line, display=False, progress_every=0
    )
    results.save_json(tmp_path / "r.json")
    results.save_csv(tmp_path / "r.csv")

    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["total_vehicles"] == 3
    assert data["counts"] == {"car": 3}
    assert len(data["events"]) == 3

    csv_text = (tmp_path / "r.csv").read_text(encoding="utf-8").strip().splitlines()
    assert csv_text[0] == "track_id,label,conf,frame_index,timestamp_s"
    assert len(csv_text) == 4


def test_pipeline_annotated_video_written(tmp_path: Path, line):
    out = tmp_path / "out.mp4"
    analyze(
        FakeDetector(),
        "synthetic",
        line,
        output_path=out,
        display=False,
        progress_every=0,
    )
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (540, 960, 3)

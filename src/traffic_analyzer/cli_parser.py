"""Argument parser for the traffic-analyzer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from .detector import DEFAULT_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic-analyzer",
        description="Detect, track and count vehicles in traffic videos using YOLO + ByteTrack.",
        epilog="Example: traffic-analyzer video.mp4 --output out.mp4 --json results.json",
    )
    parser.add_argument("source", help="Path to a video file (or a camera index, e.g. 0).")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write an annotated MP4 to this path (extension decides the codec).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write a JSON report (counts + per-vehicle events) to this path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write a CSV report (one row per counted vehicle) to this path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO weights (name or path). Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold. Default: 0.5",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Processing width (frames are resized before inference). Default: 960",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Processing height. Default: 540",
    )
    parser.add_argument(
        "--line",
        default="0.1,0.48,0.9,0.48",
        help=(
            "Counting line as x1,y1,x2,y2 in normalized [0..1] coordinates "
            "of the processing frame. Default: 0.1,0.48,0.9,0.48"
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (stride 1 = all frames). Default: 1",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a preview window (headless mode / servers).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not draw the counts panel + counting line onto --output video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run to the end). Useful for quick tests.",
    )
    return parser


def parse_line(spec: str, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert ``x1,y1,x2,y2`` normalized coords into pixel coords."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError("--line expects 4 comma-separated values: x1,y1,x2,y2")
    try:
        nx1, ny1, nx2, ny2 = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError("--line values must be numbers") from exc
    for name, v in (("x1", nx1), ("y1", ny1), ("x2", nx2), ("y2", ny2)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"--line {name}={v} outside [0,1]")
    return (
        (int(round(nx1 * width)), int(round(ny1 * height))),
        (int(round(nx2 * width)), int(round(ny2 * height))),
    )

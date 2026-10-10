"""Counting line + per-track line-crossing logic.

A vehicle is counted **once**, the first time its track crosses the line.
This replaces frame-sampled region counters, which miss objects that cross
between sampled frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

LinePoint = tuple[int, int]


def _side(p: tuple[float, float], line: tuple[LinePoint, LinePoint]) -> float:
    """Signed side of point *p* relative to the infinite line through *line*."""
    (ax, ay), (bx, by) = line
    return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)


@dataclass
class LineCrossCounter:
    """Count vehicles that cross a straight line segment.

    Crossing test: the track's center point moves from one side of the line
    to the other between consecutive frames **and** the movement segment
    properly intersects the counting line. Each track is counted at most
    once.
    """

    line: tuple[LinePoint, LinePoint]
    counted: set[int] = field(default_factory=set)
    counts: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        detections: list[dict[str, Any]],
        prev_centers: dict[int, tuple[float, float]],
        frame_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Process one frame of detections; return crossing events for it.

        ``prev_centers`` maps track_id -> center position on the previous
        frame; the caller owns this mapping and it is updated in place so
        track history survives across frames.
        """
        events: list[dict[str, Any]] = []
        for det in detections:
            tid = det["track_id"]
            cx = (det["xyxy"][0] + det["xyxy"][2]) / 2.0
            cy = (det["xyxy"][1] + det["xyxy"][3]) / 2.0
            prev = prev_centers.get(tid)
            if prev is not None and tid not in self.counted:
                if self._crossed(prev, (cx, cy)):
                    self.counted.add(tid)
                    self.counts[det["label"]] = self.counts.get(det["label"], 0) + 1
                    events.append(
                        {
                            "track_id": tid,
                            "label": det["label"],
                            "conf": det["conf"],
                            "frame_index": frame_index,
                            "position": (cx, cy),
                        }
                    )
            prev_centers[tid] = (cx, cy)

        # Prune stale tracks so the mapping cannot grow without bound.
        if len(prev_centers) > 4096:
            current = {d["track_id"] for d in detections}
            for tid in [t for t in prev_centers if t not in current]:
                del prev_centers[tid]
        return events

    def _crossed(
        self,
        prev: tuple[float, float],
        cur: tuple[float, float],
    ) -> bool:
        (ax, ay), (bx, by) = self.line
        # Bounding-box reject first (cheap).
        if max(min(prev[0], cur[0]), min(ax, bx)) > min(max(prev[0], cur[0]), max(ax, bx)):
            return False
        d1 = _side(prev, self.line)
        d2 = _side(cur, self.line)
        if d1 * d2 > 0:
            return False  # both endpoints on the same side
        # Movement segment direction, tested against both line endpoints.
        d3 = (cur[0] - prev[0]) * (ay - prev[1]) - (cur[1] - prev[1]) * (ax - prev[0])
        d4 = (cur[0] - prev[0]) * (by - prev[1]) - (cur[1] - prev[1]) * (bx - prev[0])
        return d3 * d4 < 0  # proper crossing (strict, ignores collinear grazes)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

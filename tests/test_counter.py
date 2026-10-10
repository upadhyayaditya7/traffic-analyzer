"""Unit tests for the LineCrossCounter (no ML model needed)."""

from __future__ import annotations

import pytest

from traffic_analyzer.counter import LineCrossCounter


def det(track_id: int, cx: float, cy: float, label: str = "car") -> dict:
    return {
        "track_id": track_id,
        "label": label,
        "conf": 0.9,
        "xyxy": (cx - 10, cy - 10, cx + 10, cy + 10),
    }


@pytest.fixture()
def counter() -> LineCrossCounter:
    return LineCrossCounter(line=((100, 100), (500, 100)))


def test_counts_once_when_crossing_down(counter):
    centers = {}
    counter.update([det(1, 200, 80)], centers)  # above line (y=100)
    events = counter.update([det(1, 200, 120)], centers)  # below
    assert counter.total == 1
    assert counter.counts == {"car": 1}
    assert len(events) == 1


def test_does_not_double_count(counter):
    centers = {}
    counter.update([det(1, 200, 80)], centers)
    counter.update([det(1, 200, 120)], centers)
    counter.update([det(1, 200, 160)], centers)
    assert counter.total == 1


def test_no_cross_when_staying_on_one_side(counter):
    centers = {}
    counter.update([det(1, 200, 80)], centers)
    counter.update([det(1, 240, 80)], centers)
    assert counter.total == 0


def test_cross_outside_line_segment_not_counted():
    counter = LineCrossCounter(line=((200, 100), (500, 100)))
    centers = {}
    counter.update([det(1, 50, 80)], centers)  # x=50 is left of segment start x=200
    counter.update([det(1, 50, 120)], centers)
    assert counter.total == 0


def test_per_class_counts(counter):
    centers = {}
    counter.update([det(1, 200, 80, "car"), det(2, 300, 80, "truck")], centers)
    counter.update([det(1, 200, 120, "car"), det(2, 300, 120, "truck")], centers)
    assert counter.counts == {"car": 1, "truck": 1}
    assert counter.total == 2


def test_multiple_vehicles_same_frame(counter):
    centers = {}
    counter.update([det(1, 200, 80), det(2, 300, 80), det(3, 400, 80)], centers)
    events = counter.update(
        [det(1, 200, 120), det(2, 300, 120), det(3, 400, 120)], centers
    )
    assert counter.total == 3
    assert len(events) == 3


def test_upward_crossing_also_counted(counter):
    centers = {}
    counter.update([det(1, 200, 120)], centers)  # below
    counter.update([det(1, 200, 80)], centers)  # above
    assert counter.total == 1

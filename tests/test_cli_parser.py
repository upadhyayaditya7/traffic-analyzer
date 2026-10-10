"""Tests for CLI line parsing."""

from __future__ import annotations

import pytest

from traffic_analyzer.cli_parser import parse_line


def test_default_line_pixels():
    line = parse_line("0.1,0.48,0.9,0.48", 960, 540)
    assert line == ((96, 259), (864, 259))


def test_fractional_rounding():
    line = parse_line("0.5,0.5,0.5,0.5", 100, 100)
    assert line == ((50, 50), (50, 50))


def test_rejects_three_values():
    with pytest.raises(ValueError):
        parse_line("0.1,0.2,0.3", 100, 100)


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_line("1.5,0.2,0.3,0.4", 100, 100)


def test_rejects_non_numeric():
    with pytest.raises(ValueError):
        parse_line("a,b,c,d", 100, 100)

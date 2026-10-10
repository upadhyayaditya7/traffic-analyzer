"""Deprecated shim - kept for backwards compatibility.

The packaged CLI is the way to run this project now:

    traffic-analyzer test_drone_view.mp4 --no-display --json results.json --csv events.csv
"""
from traffic_analyzer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

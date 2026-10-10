"""Command-line entry point for traffic-analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

from .cli_parser import build_parser, parse_line
from .detector import VehicleDetector
from .pipeline import analyze


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        line = parse_line(args.line, args.width, args.height)
    except ValueError as exc:
        parser.error(str(exc))

    source: str | int = args.source
    if isinstance(source, str) and source.isdigit() and not Path(source).exists():
        source = int(source)  # camera index

    print(f"Loading model: {args.model}")
    try:
        detector = VehicleDetector(model_path=args.model, conf=args.conf, iou=args.iou)
    except Exception as exc:  # pragma: no cover - depends on env
        print(f"error: could not load model {args.model!r}: {exc}", file=sys.stderr)
        return 2

    print(
        f"Analyzing {source} | processing {args.width}x{args.height} | "
        f"stride {args.stride} | line {line}"
    )

    try:
        results = analyze(
            detector,
            source,
            line,
            output_path=args.output,
            stride=args.stride,
            max_frames=args.max_frames,
            processing_size=(args.width, args.height),
            display=not args.no_display,
            draw_overlay=not args.no_overlay,
        )
    except Exception as exc:
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        return 1
    finally:
        pass

    print()
    print("=== Results ===")
    for label, n in sorted(results.counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<12} {n}")
    print(f"  {'TOTAL':<12} {results.total_vehicles}")
    print(
        f"  frames: {results.processed_frames} processed, "
        f"{results.skipped_frames} skipped | avg {results.avg_fps:.1f} fps | "
        f"{results.elapsed_seconds:.1f}s"
    )

    if args.json:
        results.save_json(args.json)
        print(f"JSON report -> {args.json}")
    if args.csv:
        results.save_csv(args.csv)
        print(f"CSV report  -> {args.csv}")
    if args.output:
        print(f"Annotated video -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

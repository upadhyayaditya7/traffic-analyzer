"""End-to-end analysis pipeline: video -> detections -> crossings -> outputs."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .counter import LineCrossCounter
from .detector import VehicleDetector, VideoSourceError
from .reporting import CrossEvent, Overlay, Results


def analyze(
    detector: VehicleDetector,
    source: str | Path | int,
    line: tuple[tuple[int, int], tuple[int, int]],
    *,
    output_path: str | Path | None = None,
    draw_overlay: bool = True,
    draw_detections: bool = True,
    stride: int = 1,
    max_frames: int = 0,
    processing_size: tuple[int, int] = (960, 540),
    display: bool = True,
    progress_every: int = 100,
) -> Results:
    """Run detection + counting over *source* and return aggregate results.

    Every frame is read (nothing is skipped for counting correctness);
    ``stride`` only controls how often *inference* runs, so tracks still get
    centers interpolated... actually centers are only updated on processed
    frames, so large strides can still miss crossings -- stride 1 is the
    default and the accurate mode.
    """
    stride = max(1, int(stride))
    results = Results(
        source=str(source),
        model=detector.model.model_name if hasattr(detector.model, "model_name") else str(detector.model),
        processed_width=processing_size[0],
        processed_height=processing_size[1],
    )

    counter = LineCrossCounter(line=line)
    overlay = Overlay(line) if draw_overlay else None

    writer: cv2.VideoWriter | None = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    prev_centers: dict[int, tuple[float, float]] = {}
    t_start = time.perf_counter()

    try:
        for chunk in detector.track_frames(source, *processing_size):
            frame = chunk["frame"]
            dets = chunk["detections"]
            idx = chunk["frame_index"]
            results.total_frames = idx

            if writer is None and output_path is not None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    20.0,  # nominal output fps; annotated preview video
                    (frame.shape[1], frame.shape[0]),
                )

            if idx % stride == 0:
                events = counter.update(dets, prev_centers, frame_index=idx)
                results.processed_frames += 1
                for ev in events:
                    results.events.append(
                        CrossEvent(
                            track_id=ev["track_id"],
                            label=ev["label"],
                            conf=ev["conf"],
                            frame_index=ev["frame_index"] or idx,
                            fps_at_event=chunk["fps"],
                        )
                    )
                results.counts = dict(counter.counts)
            else:
                results.skipped_frames += 1

            if (overlay is not None or writer is not None or display) and idx % stride == 0:
                canvas = frame
                if draw_detections and overlay is not None:
                    canvas = Overlay.draw_detections(canvas, dets)
                if overlay is not None:
                    canvas = overlay.draw(
                        canvas,
                        dets,
                        counter.counts,
                        counter.total,
                        chunk["fps"],
                    )
                if writer is not None:
                    writer.write(canvas)
                if display:
                    cv2.imshow("Traffic Analyzer", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Interrupted by user ('q').")
                        break

            if progress_every and idx % progress_every == 0:
                print(
                    f"  frame {idx:5d} | tracked {len(dets):2d} | "
                    f"counted {counter.total:4d} | {chunk['fps']:6.1f} fps",
                    flush=True,
                )
            if max_frames and idx >= max_frames:
                break
    except VideoSourceError:
        raise
    finally:
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    results.elapsed_seconds = time.perf_counter() - t_start
    results.avg_fps = (
        results.processed_frames / results.elapsed_seconds
        if results.elapsed_seconds > 0
        else 0.0
    )
    results.counts = dict(counter.counts)
    return results

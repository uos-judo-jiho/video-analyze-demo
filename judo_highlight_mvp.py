import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "yolov8n-pose.pt"


@dataclass
class FallEvent:
    time_sec: float
    confidence: float
    aspect_ratio: float
    body_angle_deg: float | None
    start_sec: float = 0.0
    end_sec: float = 0.0


def body_angle_deg(keypoints: np.ndarray) -> float | None:
    """Estimate torso angle from shoulder and hip keypoints."""
    if keypoints.shape[0] < 13:
        return None

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    if np.any(left_shoulder[:2] <= 0) or np.any(right_shoulder[:2] <= 0):
        return None
    if np.any(left_hip[:2] <= 0) or np.any(right_hip[:2] <= 0):
        return None

    shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
    hip_center = (left_hip[:2] + right_hip[:2]) / 2

    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]

    return abs(float(np.degrees(np.arctan2(dy, dx))))


def horizontal_distance_deg(angle_deg: float | None) -> float | None:
    if angle_deg is None:
        return None
    return min(abs(angle_deg), abs(180.0 - angle_deg))


def score_pose(box: np.ndarray, keypoints: np.ndarray) -> tuple[bool, float, float, float | None]:
    x1, y1, x2, y2 = box
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    aspect_ratio = width / height
    angle_deg = body_angle_deg(keypoints)
    horizontal_deg = horizontal_distance_deg(angle_deg)

    score = 0.0
    if aspect_ratio > 1.15:
        score += 0.45

    if horizontal_deg is not None and horizontal_deg < 35.0:
        score += 0.45

    return score >= 0.65, score, aspect_ratio, angle_deg


def merge_events(events: list[FallEvent], gap_sec: float = 8.0) -> list[FallEvent]:
    if not events:
        return []

    ordered = sorted(events, key=lambda event: event.time_sec)
    merged = [ordered[0]]

    for event in ordered[1:]:
        previous = merged[-1]
        if event.time_sec - previous.time_sec <= gap_sec:
            if event.confidence > previous.confidence:
                merged[-1] = event
        else:
            merged.append(event)

    return merged


def cut_clip(input_path: Path, output_path: Path, start_sec: float, end_sec: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def get_video_duration(input_path: Path) -> float:
    capture = cv2.VideoCapture(str(input_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    if fps <= 0:
        raise RuntimeError("Unable to read video FPS.")

    return float(frame_count / fps)


def save_events_json(events: list[FallEvent], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.json"
    payload = {"events": [asdict(event) for event in events]}
    events_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return events_path


def analyze_video(
    input_path: Path,
    output_dir: Path,
    sample_fps: float = 5.0,
    before_sec: float = 10.0,
    after_sec: float = 10.0,
) -> list[FallEvent]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than 0.")
    if before_sec < 0 or after_sec < 0:
        raise ValueError("--before and --after must be 0 or greater.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available on PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)

    capture = cv2.VideoCapture(str(input_path))
    original_fps = capture.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        capture.release()
        raise RuntimeError("Unable to read video FPS.")

    duration_sec = get_video_duration(input_path)
    frame_interval = max(1, int(round(original_fps / sample_fps)))
    events: list[FallEvent] = []
    frame_idx = 0

    print(f"Start analysis: input={input_path} sample_fps={sample_fps} duration={duration_sec:.2f}s")

    while True:
        has_frame, frame = capture.read()
        if not has_frame:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        time_sec = frame_idx / original_fps
        results = model(frame, verbose=False)

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy()

            for person_idx, (box, kpts) in enumerate(zip(boxes, keypoints), start=1):
                fall_like, confidence, aspect_ratio, angle_deg = score_pose(box, kpts)
                angle_label = "None" if angle_deg is None else f"{angle_deg:.2f}"

                print(
                    f"[frame={frame_idx:06d} t={time_sec:7.2f}s person={person_idx}] "
                    f"aspect_ratio={aspect_ratio:.3f} body_angle_deg={angle_label} "
                    f"confidence={confidence:.2f} fall_like={fall_like}"
                )

                if fall_like:
                    events.append(
                        FallEvent(
                            time_sec=float(time_sec),
                            confidence=float(confidence),
                            aspect_ratio=float(aspect_ratio),
                            body_angle_deg=None if angle_deg is None else float(angle_deg),
                        )
                    )

        frame_idx += 1

    capture.release()

    merged_events = merge_events(events)
    print(f"Detected events: raw={len(events)} merged={len(merged_events)}")

    for index, event in enumerate(merged_events, start=1):
        event.start_sec = max(0.0, event.time_sec - before_sec)
        event.end_sec = min(duration_sec, event.time_sec + after_sec)
        output_path = output_dir / f"highlight_{index:02d}_{int(event.start_sec)}_{int(event.end_sec)}.mp4"
        print(
            f"Cut clip #{index}: event={event.time_sec:.2f}s "
            f"range={event.start_sec:.2f}s~{event.end_sec:.2f}s output={output_path}"
        )
        cut_clip(input_path, output_path, event.start_sec, event.end_sec)

    events_path = save_events_json(merged_events, output_dir)
    print(f"Saved events metadata: {events_path}")
    print("Done")
    return merged_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect fall-like judo moments and cut highlight clips."
    )
    parser.add_argument("input", help="Path to the input mp4 video.")
    parser.add_argument("--output", default="outputs", help="Directory for clips and events.json.")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Sampling FPS for analysis.")
    parser.add_argument("--before", type=float, default=10.0, help="Seconds before each event.")
    parser.add_argument("--after", type=float, default=10.0, help="Seconds after each event.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyze_video(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        sample_fps=args.sample_fps,
        before_sec=args.before,
        after_sec=args.after,
    )


if __name__ == "__main__":
    main()

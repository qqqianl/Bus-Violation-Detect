"""Create an MP4 video from image frames sorted by their final numeric token."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2


def numeric_key(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[-1]) if numbers else -1, path.name)


def create_video(frame_dir: Path, output: Path, fps: float = 25.0) -> int:
    frames = sorted(
        (p for p in frame_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=numeric_key,
    )
    if not frames:
        raise ValueError(f"no image frames found in {frame_dir}")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"cannot read first frame: {frames[0]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    written = 0
    try:
        for path in frames:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()
    print(f"wrote {create_video(args.frame_dir, args.output, args.fps)} frame(s)")


"""Train an Ultralytics YOLO detector from an external dataset YAML file."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="YOLO dataset YAML (not included)")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="road_scene")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    YOLO(args.model).train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        project=args.project,
        name=args.name,
    )


"""Track one person and save square crops for the 3D solver."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.track import SubjectTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, or 0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--vram-limit-mb", type=int, default=3072)
    args = parser.parse_args()
    SubjectTracker(args.model, args.device, args.imgsz, args.crop_size,
                   args.vram_limit_mb, args.tracker).run(args.video, args.run)


if __name__ == "__main__":
    main()

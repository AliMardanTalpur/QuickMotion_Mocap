"""Stage 1: extract one COCO-17 actor from an MP4 with YOLOv8 pose.

Output is an NPY array (frames, 17, 3) in pixel x/y/confidence order, plus a
JSON sidecar with video metadata. Missing joints/frames are NaN x/y and zero confidence. No interpolation
occurs until Stage 2, so the original detections remain inspectable.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def choose_actor(boxes: np.ndarray, previous: np.ndarray | None) -> int:
    """Prefer the detection nearest the previous actor; start with top confidence."""
    if previous is None:
        return int(np.argmax(boxes[:, 4]))
    centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
    old_center = (previous[:2] + previous[2:4]) / 2
    old_size = max(np.linalg.norm(previous[2:4] - previous[:2]), 1.0)
    distance = np.linalg.norm(centers - old_center, axis=1) / old_size
    size = np.linalg.norm(boxes[:, 2:4] - boxes[:, :2], axis=1)
    size_change = np.abs(np.log(np.maximum(size, 1) / max(old_size, 1)))
    score = boxes[:, 4] - 0.7 * distance - 0.15 * size_change
    return int(np.argmax(score))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input MP4")
    parser.add_argument("--output", type=Path, default=Path("2d_keypoints.npy"))
    parser.add_argument("--model", default="yolov8n-pose.pt",
                        help="Use yolov8s-pose.pt for higher accuracy when its weights are available")
    parser.add_argument("--device", default="auto", help="auto, cpu, or 0 for CUDA GPU 0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--joint-conf", type=float, default=0.2)
    parser.add_argument("--recover-white-shoes", action="store_true",
                        help="Static-camera correction for white shoes in acrobatic clips")
    args = parser.parse_args()

    if not args.video.is_file() or args.video.suffix.lower() != ".mp4":
        parser.error("video must be an existing .mp4 file")
    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError("Video FPS is unavailable; re-encode the MP4 with a valid FPS")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video dimensions are invalid")

    model = YOLO(args.model)
    frames: list[np.ndarray] = []
    previous_box: np.ndarray | None = None
    missing = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pose = np.full((17, 3), np.nan, dtype=np.float32)
            pose[:, 2] = 0.0
            # One frame at a time; inference_mode avoids autograd GPU allocations.
            with torch.inference_mode():
                result = model.predict(
                    source=frame, device=device, imgsz=args.imgsz,
                    conf=args.person_conf, classes=[0], verbose=False,
                )[0]
            if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
                boxes = np.concatenate(
                    (result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()[:, None]),
                    axis=1,
                )
                if len(result.keypoints.xy) != len(boxes):
                    raise RuntimeError("Detector boxes and keypoints are misaligned")
                selected = choose_actor(boxes, previous_box)
                xy = result.keypoints.xy[selected].cpu().numpy()
                conf_tensor = result.keypoints.conf
                confidence = (
                    conf_tensor[selected].cpu().numpy()
                    if conf_tensor is not None else np.ones(17, dtype=np.float32)
                )
                if xy.shape != (17, 2) or confidence.shape != (17,):
                    raise RuntimeError("Expected a COCO-17 YOLO pose model")
                valid = np.isfinite(xy).all(axis=1) & (confidence >= args.joint_conf)
                pose[valid, :2] = xy[valid]
                pose[valid, 2] = confidence[valid]
                previous_box = boxes[selected].copy()
            else:
                missing += 1
            del result
            frames.append(pose)
            if len(frames) % 100 == 0:
                print(f"Stage 1: {len(frames)} frames", flush=True)
    finally:
        cap.release()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not frames:
        raise RuntimeError("No frames were decoded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sequence = np.stack(frames)
    recovery_report = {}
    if args.recover_white_shoes:
        # Runs after `model` is deleted, so this CPU image pass cannot overlap
        # the pose network in the 4 GB GPU memory budget.
        from white_shoe_recovery import recover_white_shoes
        sequence, recovery_report = recover_white_shoes(args.video, sequence)
        print("Shoe recovery:", recovery_report)
    np.save(args.output, sequence)
    args.output.with_suffix(".json").write_text(
        json.dumps({"fps": fps, "width": width, "height": height,
                    "joint_order": "COCO-17", "model": args.model,
                    **recovery_report}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(frames)} frames to {args.output}; {missing} missed detections")


if __name__ == "__main__":
    main()

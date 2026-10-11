"""YOLOv8 pose + ByteTrack pass, then one actor's square crop pass."""

from __future__ import annotations

from collections import defaultdict
import gc
from pathlib import Path

import cv2
import numpy as np

from .contracts import TrackSequence
from .crop import crop_square, square_roi
from .gpu_budget import check, configure


class SubjectTracker:
    def __init__(self, model_path: str, device: str = "auto", imgsz: int = 960,
                 crop_size: int = 512, vram_limit_mb: int = 3072,
                 tracker_config: str = "bytetrack.yaml") -> None:
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.crop_size = crop_size
        self.vram_limit_mb = vram_limit_mb
        self.tracker_config = tracker_config

    def run(self, video: Path, run_dir: Path) -> Path:
        import torch
        from ultralytics import YOLO

        if not video.is_file():
            raise FileNotFoundError(video)
        run_dir.mkdir(parents=True, exist_ok=True)
        crop_dir = run_dir / "crops"
        crop_dir.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not read {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Invalid video FPS or dimensions")
        device = ("0" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        configure(torch, device, self.vram_limit_mb)
        model = YOLO(self.model_path)
        observations: list[dict[int, tuple[np.ndarray, np.ndarray, float]]] = []
        scores: dict[int, list[float]] = defaultdict(list)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                with torch.inference_mode():
                    pred = model.track(
                        frame, persist=True, tracker=self.tracker_config,
                        classes=[0], imgsz=self.imgsz, conf=.15,
                        device=device, verbose=False,
                    )[0]
                current = {}
                boxes = pred.boxes
                if boxes is not None and boxes.id is not None and pred.keypoints is not None:
                    ids = boxes.id.int().cpu().numpy()
                    xyxy = boxes.xyxy.cpu().numpy()
                    box_conf = boxes.conf.cpu().numpy()
                    xy = pred.keypoints.xy.cpu().numpy()
                    kp_conf = (pred.keypoints.conf.cpu().numpy()
                               if pred.keypoints.conf is not None
                               else np.ones((len(ids), 17), dtype=np.float32))
                    for j, actor_id in enumerate(ids):
                        kp = np.column_stack((xy[j], kp_conf[j])).astype(np.float32)
                        kp[kp[:, 2] < .1, :2] = np.nan
                        actor_id = int(actor_id)
                        current[actor_id] = (xyxy[j].astype(np.float32), kp, float(box_conf[j]))
                        scores[actor_id].append(float(box_conf[j]))
                observations.append(current)
                if len(observations) % 100 == 0:
                    print(f"Tracked {len(observations)} frames", flush=True)
                if device != "cpu":
                    check(torch, device, self.vram_limit_mb)
        finally:
            cap.release()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not observations or not scores:
            raise RuntimeError("No persistent person track was found")
        # Longest observed ID wins; confidence breaks ties. A single actor is
        # selected for the complete video, never reselected per frame.
        actor_id = max(scores, key=lambda x: (len(scores[x]), np.mean(scores[x])))
        n = len(observations)
        bbox = np.full((n, 4), np.nan, np.float32)
        keypoints = np.full((n, 17, 3), np.nan, np.float32)
        keypoints[:, :, 2] = 0
        valid = np.zeros(n, bool)
        confidence = np.zeros(n, np.float32)
        for i, visible in enumerate(observations):
            if actor_id in visible:
                bbox[i], keypoints[i], confidence[i] = visible[actor_id]
                valid[i] = True
        crop_valid = valid.copy()
        # Bridge only brief gaps for solver continuity; keep valid=False so
        # downstream quality checks can distinguish inferred from observed.
        observed_indices = np.flatnonzero(valid)
        for start, end in zip(observed_indices[:-1], observed_indices[1:]):
            if 1 < end - start <= 6:
                for i in range(start + 1, end):
                    alpha = (i - start) / (end - start)
                    bbox[i] = (1 - alpha) * bbox[start] + alpha * bbox[end]
                    crop_valid[i] = True
        roi = np.full_like(bbox, np.nan)
        transforms = np.full((n, 3, 3), np.nan, np.float32)
        cap = cv2.VideoCapture(str(video))
        try:
            for i in range(n):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Video ended early during crop pass at frame {i}")
                if not crop_valid[i]:
                    continue
                roi[i] = square_roi(bbox[i], keypoints[i])
                crop, transforms[i] = crop_square(frame, roi[i], self.crop_size)
                if not cv2.imwrite(str(crop_dir / f"{i:06d}.png"), crop):
                    raise RuntimeError(f"Could not write crop for frame {i}")
        finally:
            cap.release()
        output = run_dir / "tracking.npz"
        TrackSequence(
            frame_index=np.arange(n, dtype=np.int32),
            track_id=np.where(valid, actor_id, -1).astype(np.int32),
            bbox_xyxy=bbox, crop_xyxy=roi, crop_to_image=transforms,
            keypoints_xyc=keypoints, valid=valid, crop_valid=crop_valid,
        ).save(output, video=str(video.resolve()), fps=fps, width=width,
               height=height, crop_size=self.crop_size, actor_id=actor_id,
               tracker=self.tracker_config, model=self.model_path,
               observed_frames=int(valid.sum()), crop_frames=int(crop_valid.sum()))
        print(f"Saved {output}: {valid.sum()}/{n} observed frames, actor ID {actor_id}")
        return output

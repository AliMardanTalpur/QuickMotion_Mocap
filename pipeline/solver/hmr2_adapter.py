"""Run the official 4DHumans HMR2 model on tracked crops, one at a time.

Requires a separately installed official 4DHumans checkout, its checkpoint,
and its licensed SMPL_NEUTRAL body model. No model is downloaded at runtime.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from ..contracts import SmplSequence, TrackSequence, read_metadata
from ..gpu_budget import check, configure


class HMR2Solver:
    def __init__(self, repo: Path, checkpoint: Path, device: str = "cpu",
                 vram_limit_mb: int = 3072) -> None:
        self.repo = repo.resolve()
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.vram_limit_mb = vram_limit_mb

    def solve(self, tracking_file: Path, output_file: Path) -> Path:
        if not self.repo.is_dir() or not (self.repo / "hmr2").is_dir():
            raise FileNotFoundError("--hmr2-repo must point to the official 4DHumans checkout")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"HMR2 checkpoint is missing: {self.checkpoint}")
        import torch

        configure(torch, self.device, self.vram_limit_mb)
        # The upstream renderer defaults to Linux EGL even though rendering is
        # unused by this inference adapter. Select native Windows OpenGL first.
        if os.name == "nt":
            os.environ.setdefault("PYOPENGL_PLATFORM", "win32")
        sys.path.insert(0, str(self.repo))
        try:
            from hmr2.models import load_hmr2
            from hmr2.datasets.vitdet_dataset import ViTDetDataset
        except ImportError as exc:
            raise RuntimeError(
                "The official 4DHumans dependencies are unavailable in this Python environment. "
                "Install its environment per the 4DHumans README and run this command with that Python."
            ) from exc

        tracking = TrackSequence.load(tracking_file)
        meta = read_metadata(tracking_file)
        n = len(tracking.frame_index)
        if not tracking.crop_valid.any():
            raise RuntimeError("No tracked crops are available")
        device = torch.device("cuda:0" if self.device != "cpu" else "cpu")
        # load_hmr2 checks that the user-supplied SMPL model is installed.
        model, config = load_hmr2(str(self.checkpoint))
        model = model.to(device).eval()
        check(torch, self.device, self.vram_limit_mb)
        rotations = np.full((n, 24, 3, 3), np.nan, np.float32)
        per_frame_betas = np.full((n, 10), np.nan, np.float32)
        translation = np.full((n, 3), np.nan, np.float32)
        joints = np.full((n, 24, 3), np.nan, np.float32)
        valid = np.zeros(n, bool)
        confidence = np.zeros(n, np.float32)
        peak_mb = 0.0
        try:
            for i in range(n):
                if not tracking.crop_valid[i]:
                    continue
                crop = cv2.imread(str(tracking_file.parent / "crops" / f"{i:06d}.png"))
                if crop is None:
                    raise RuntimeError(f"Missing crop for frame {i}")
                h, w = crop.shape[:2]
                dataset = ViTDetDataset(config, crop,
                                        np.array([[0, 0, w, h]], np.float32))
                sample = dataset[0]
                batch = {"img": torch.as_tensor(sample["img"])[None].to(device)}
                with torch.inference_mode():
                    out = model(batch)
                params = out["pred_smpl_params"]
                orient = params["global_orient"].reshape(1, 1, 3, 3)
                pose = params["body_pose"].reshape(1, 23, 3, 3)
                rotations[i] = torch.cat((orient, pose), dim=1)[0].float().cpu().numpy()
                per_frame_betas[i] = params["betas"][0, :10].float().cpu().numpy()
                # Official HMR2 camera-crop-to-full formula. The dataset may
                # expand its crop internally; map its box size back through
                # our saved square ROI before using the original frame size.
                cam = out["pred_cam"][0].float().cpu().numpy()
                roi = tracking.crop_xyxy[i]
                center = (roi[:2] + roi[2:]) / 2
                bbox_size = (float(sample["box_size"]) / w) * (roi[2] - roi[0])
                focal = (float(config.EXTRA.FOCAL_LENGTH) /
                         float(config.MODEL.IMAGE_SIZE) *
                         max(int(meta["width"]), int(meta["height"])))
                scaled = bbox_size * cam[0] + 1e-9
                tx = 2 * (center[0] - int(meta["width"]) / 2) / scaled + cam[1]
                ty = 2 * (center[1] - int(meta["height"]) / 2) / scaled + cam[2]
                tz = 2 * focal / scaled
                translation[i] = (tx, ty, tz)
                local_joints = out["pred_keypoints_3d"][0, :24].float().cpu().numpy()
                joints[i] = local_joints + translation[i]
                valid[i] = bool(tracking.valid[i])
                kp_conf = tracking.keypoints_xyc[i, :, 2]
                confidence[i] = float(np.nanmean(kp_conf)) if tracking.valid[i] else 0.0
                del out, params, sample, dataset, batch
                if self.device != "cpu":
                    peak_mb = max(peak_mb, check(torch, self.device, self.vram_limit_mb))
                if (i + 1) % 25 == 0:
                    print(f"HMR2: {i + 1}/{n} frames", flush=True)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not (np.isfinite(rotations[tracking.crop_valid]).all()
                and np.isfinite(translation[tracking.crop_valid]).all()
                and np.isfinite(joints[tracking.crop_valid]).all()
                and np.isfinite(per_frame_betas[tracking.crop_valid]).all()):
            raise RuntimeError("HMR2 returned non-finite SMPL predictions")
        stable_betas = np.median(per_frame_betas[tracking.crop_valid], axis=0)
        result = SmplSequence(rotations, stable_betas.astype(np.float32), translation,
                              joints, valid, confidence)
        result.save(
            output_file, model="HMR2/4DHumans", body_model="SMPL-24",
            fps=float(meta["fps"]), source_tracking=str(tracking_file.resolve()),
            coordinate_system="HMR2 camera estimate; world trajectory uncalibrated",
            ground_aligned=False, vram_peak_reserved_mb=peak_mb,
            device=str(device), batch_size=1,
        )
        print(f"Saved {output_file}; peak PyTorch reserved VRAM {peak_mb:.0f} MB")
        return output_file

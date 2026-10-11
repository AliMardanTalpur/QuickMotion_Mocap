"""Stage 2: lift COCO-17 2D tracks with the official 243-frame VideoPose3D model.

The pretrained model emits Human3.6M-17 camera-space, pelvis-relative meters.
It was trained on Detectron COCO detections; YOLO detections are compatible in
joint order but have a detector domain shift. Screen-space root motion is only
an optional image-plane approximation, not monocular absolute depth recovery.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch import nn

from motion_processing import clean_2d, clean_3d, gaussian_time, regularize_front_depth
from pose_reconstruction import model_reprojection_error, reconstruct_from_2d


CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/video-pose-3d/"
    "pretrained_h36m_detectron_coco.bin"
)
JOINTS_H36M = [
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "spine", "thorax",
    "neck", "head", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]


class TemporalModel(nn.Module):
    """State-dict-compatible inference form of VideoPose3D's temporal network."""

    def __init__(self, channels: int = 1024) -> None:
        super().__init__()
        widths = [3, 3, 3, 3, 3]
        self.expand_conv = nn.Conv1d(34, channels, 3, bias=False)
        self.expand_bn = nn.BatchNorm1d(channels)
        self.layers_conv = nn.ModuleList()
        self.layers_bn = nn.ModuleList()
        dilation = 3
        self.pads: list[int] = []
        for width in widths[1:]:
            self.pads.append((width - 1) * dilation // 2)
            self.layers_conv.extend([
                nn.Conv1d(channels, channels, width, dilation=dilation, bias=False),
                nn.Conv1d(channels, channels, 1, bias=False),
            ])
            self.layers_bn.extend([
                nn.BatchNorm1d(channels), nn.BatchNorm1d(channels),
            ])
            dilation *= width
        self.shrink = nn.Conv1d(channels, 51, 1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.25)

    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        if poses.ndim != 4 or poses.shape[2:] != (17, 2):
            raise ValueError("Expected (batch, frames, 17, 2)")
        x = poses.reshape(poses.shape[0], poses.shape[1], 34).transpose(1, 2)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for block, pad in enumerate(self.pads):
            residual = x[:, :, pad:-pad]
            x = self.drop(self.relu(self.layers_bn[2 * block](
                self.layers_conv[2 * block](x)
            )))
            x = residual + self.drop(self.relu(self.layers_bn[2 * block + 1](
                self.layers_conv[2 * block + 1](x)
            )))
        x = self.shrink(x).transpose(1, 2)
        return x.reshape(poses.shape[0], -1, 17, 3)


def interpolate_missing(keypoints: np.ndarray) -> np.ndarray:
    xy = keypoints[:, :, :2].copy().astype(np.float32)
    timeline = np.arange(len(xy))
    observed = (keypoints[:, :, 2] > 0) & np.isfinite(xy).all(axis=2)
    if not observed.any():
        raise ValueError("No valid 2D keypoints were detected in this clip")
    unavailable: list[int] = []
    for joint in range(17):
        valid = observed[:, joint]
        if not valid.any():
            unavailable.append(joint)
            continue
        for axis in range(2):
            xy[:, joint, axis] = np.interp(
                timeline, timeline[valid], xy[valid, joint, axis]
            )
    # A completely invisible joint cannot be interpolated. Use a neighboring
    # visible joint trajectory as a degraded but finite input to the lifter.
    neighbors = [5, 0, 0, 1, 2, 6, 5, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
    available = [joint for joint in range(17) if joint not in unavailable]
    for joint in unavailable:
        fallback = neighbors[joint] if neighbors[joint] in available else available[0]
        xy[:, joint] = xy[:, fallback]
        print(f"Warning: joint {joint} never observed; copied joint {fallback}")
    return xy


def normalize_screen(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    # Exact coordinate convention of the official VideoPose3D checkpoint.
    return xy / width * 2.0 - np.array([1.0, height / width], dtype=np.float32)


def load_model(checkpoint_path: Path, device: torch.device) -> TemporalModel:
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading official VideoPose3D checkpoint to {checkpoint_path}")
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".part")
        try:
            urlretrieve(CHECKPOINT_URL, temporary)
            temporary.replace(checkpoint_path)
        finally:
            temporary.unlink(missing_ok=True)
    # PyTorch's legacy VideoPose3D archive includes non-tensor metadata. Loading
    # it requires pickle; use only this official URL or a checkpoint you trust.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_pos"]
    channels = int(state["expand_conv.weight"].shape[0])
    model = TemporalModel(channels)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def lift(
    model: TemporalModel, xy: np.ndarray, device: torch.device, chunk: int,
    tta: bool,
) -> np.ndarray:
    # Each output frame has 121 preceding and 121 succeeding frames. Processing
    # small output chunks keeps activation memory bounded on a 4 GB GPU.
    padded = np.pad(xy, ((121, 121), (0, 0), (0, 0)), mode="edge")
    output = np.empty((len(xy), 17, 3), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(xy), chunk):
            end = min(start + chunk, len(xy))
            window = torch.from_numpy(padded[start:end + 242][None]).to(device)
            prediction = model(window)[0].cpu().numpy()
            if tta:
                # Official horizontal flip augmentation, evaluated sequentially
                # so peak GPU use is one model and one window at a time.
                left_2d = [1, 3, 5, 7, 9, 11, 13, 15]
                right_2d = [2, 4, 6, 8, 10, 12, 14, 16]
                flipped = window.clone()
                flipped[..., 0] *= -1
                flipped[:, :, left_2d + right_2d] = flipped[:, :, right_2d + left_2d].clone()
                mirrored = model(flipped)[0].cpu().numpy()
                left_3d = [4, 5, 6, 11, 12, 13]
                right_3d = [1, 2, 3, 14, 15, 16]
                mirrored[:, left_3d + right_3d] = mirrored[:, right_3d + left_3d].copy()
                mirrored[..., 0] *= -1
                prediction = (prediction + mirrored) / 2
                del flipped
            if prediction.shape != (end - start, 17, 3):
                raise RuntimeError(f"Unexpected model output {prediction.shape}")
            output[start:end] = prediction
            del window
    output -= output[:, :1, :]
    return output


def approximate_image_root(output: np.ndarray, xy: np.ndarray) -> None:
    # COCO hip midpoint is a usable 2D root. Convert pixel displacement to an
    # approximate image-plane metric displacement using body height.
    hip = (xy[:, 11] + xy[:, 12]) / 2
    head = xy[:, 0]
    ankles = (xy[:, 15] + xy[:, 16]) / 2
    pixel_height = np.median(np.linalg.norm(head - ankles, axis=1))
    model_height = np.median(np.linalg.norm(output[:, 10] - output[:, 3], axis=1))
    if pixel_height < 10 or model_height < 0.3:
        raise RuntimeError("Body scale cannot be estimated for screen root motion")
    translation = (hip - hip[0]) * (model_height / pixel_height)
    output[:, :, 0] += translation[:, None, 0]
    output[:, :, 1] += translation[:, None, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("2d_keypoints.npy"))
    parser.add_argument("--output", type=Path, default=Path("3d_points.npy"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/pretrained_h36m_detectron_coco.bin"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--chunk", type=int, default=32, help="Output frames per forward pass")
    parser.add_argument("--root-motion", choices=("fixed", "screen"), default="screen")
    parser.add_argument("--no-tta", action="store_true", help="Disable sequential flip augmentation")
    parser.add_argument("--no-smooth", action="store_true", help="Disable 2D/3D cleanup")
    parser.add_argument("--target-height", type=float, default=1.65,
                        help="Target head-to-ankle distance in meters; 0 preserves checkpoint scale")
    parser.add_argument("--depth-scale", type=float, default=1.0,
                        help="Scale relative camera depth; values below 1 reduce uncertain depth motion")
    parser.add_argument("--depth-prior", choices=("front", "model", "planar"), default="front",
                        help="Use front for ordinary footage, planar for side-on floor work, or model")
    parser.add_argument("--pose-mode", choices=("auto", "model", "image-guided"), default="auto",
                        help="Use 2D-guided geometry when the model disagrees with visible joints")
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("--chunk must be positive")
    if args.depth_scale <= 0:
        parser.error("--depth-scale must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    keypoints = np.load(args.input, allow_pickle=False)
    source_metadata = json.loads(args.input.with_suffix(".json").read_text(encoding="utf-8"))
    fps = float(source_metadata["fps"])
    width, height = int(source_metadata["width"]), int(source_metadata["height"])
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3) or len(keypoints) == 0:
        raise ValueError("Stage 1 input must have shape (frames, 17, 3)")
    xy_pixels = interpolate_missing(keypoints)
    if not args.no_smooth:
        xy_pixels = clean_2d(xy_pixels, keypoints[:, :, 2])
        if source_metadata.get("white_shoe_recovery"):
            # Image-derived ankle identities can switch over a few frames in
            # fast floor work. Smooth only the corrected leg tracks before
            # lifting; shoulders, head, and hands retain their usual timing.
            xy_pixels[:, 13:17] = gaussian_time(xy_pixels[:, 13:17], sigma=1.3)
    xy = normalize_screen(xy_pixels, width, height).astype(np.float32)
    model = load_model(args.checkpoint, device)
    try:
        points = lift(model, xy, device, args.chunk, tta=not args.no_tta)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not args.no_smooth:
        points = clean_3d(points)
    reprojection_error = model_reprojection_error(points, keypoints)
    pose_mode = args.pose_mode
    if pose_mode == "auto":
        pose_mode = "image-guided" if reprojection_error > 0.10 else "model"
    pose_report = {}
    if pose_mode == "image-guided":
        points, pose_report = reconstruct_from_2d(
            xy_pixels, points, args.target_height if args.target_height > 0 else 1.65,
            depth_prior=args.depth_prior,
        )
    else:
        if args.depth_prior == "front":
            points = regularize_front_depth(points)
        if args.target_height > 0:
            reference_height = float(np.median(np.linalg.norm(
                points[:, 10] - (points[:, 3] + points[:, 6]) / 2, axis=1
            )))
            if reference_height < 0.3:
                raise RuntimeError("Lifter produced an implausibly short skeleton")
            points *= args.target_height / reference_height
    points[:, :, 2] *= args.depth_scale
    if args.root_motion == "screen" and pose_mode == "model":
        approximate_image_root(points, xy_pixels)
    if args.root_motion == "fixed" and pose_mode == "image-guided":
        points[:, :, :2] -= points[:, :1, :2]
    if not np.isfinite(points).all():
        raise RuntimeError("Non-finite 3D predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, points)
    metadata = {
        "fps": fps, "joint_order": JOINTS_H36M, "units": "meters",
        "coordinates": "camera: x right, y down, z forward",
        "root_motion": args.root_motion,
        "flip_augmentation": not args.no_tta,
        "temporal_cleanup": not args.no_smooth,
        "target_height": args.target_height,
        "depth_scale": args.depth_scale,
        "depth_prior": args.depth_prior,
        "pose_mode": pose_mode,
        "model_reprojection_relative_error": reprojection_error,
        **pose_report,
        "checkpoint": str(args.checkpoint),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {points.shape} to {args.output} ({device})")


if __name__ == "__main__":
    main()

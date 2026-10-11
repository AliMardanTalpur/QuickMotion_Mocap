"""Run the optional GVHMR CPU route and prepare joints for Blender export.

Install the maintained GVHMR port in a separate Python environment first.
Its registered SMPL-X body model and four published checkpoints are required.
This wrapper runs preprocessing and motion recovery entirely on CPU, so the
RTX 2050's 4 GB VRAM is never shared by two GVHMR models.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from gvhmr_bridge import save_sequence


WEIGHTS = {
    "gvhmr/gvhmr_siga24_release.ckpt": 163_508_011,
    "hmr2/epoch=10-step=25000.ckpt": 2_709_494_041,
    "vitpose/vitpose-h-multi-coco.pth": 2_549_075_546,
    "yolo/yolov8x.pt": 136_867_539,
}


def missing_assets(repo: Path) -> list[str]:
    checkpoint_root = Path(os.environ.get("GVHMR_CHECKPOINTS", repo / "inputs/checkpoints"))
    body_root = Path(os.environ.get("GVHMR_BODY_MODELS", checkpoint_root / "body_models"))
    missing = []
    smplx = body_root / "smplx/SMPLX_NEUTRAL.npz"
    if not smplx.is_file():
        missing.append(f"licensed SMPL-X model: {smplx}")
    for relative, minimum_size in WEIGHTS.items():
        file = checkpoint_root / relative
        if not file.is_file() or file.stat().st_size != minimum_size:
            missing.append(f"checkpoint: {file}")
    return missing


def smplx_camera_joints(params: dict, batch_size: int = 16) -> np.ndarray:
    """Evaluate the licensed SMPL-X body model without requiring SMPL files."""
    import torch
    from gvhmr.utils.smplx_utils import make_smplx

    length = len(params["transl"])
    if not length or batch_size < 1:
        raise ValueError("SMPL-X parameters and batch size must be nonempty")
    body = make_smplx("supermotion").to("cpu").eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, length, batch_size):
            end = min(start + batch_size, length)
            batch = {key: value[start:end].detach().cpu()
                     for key, value in params.items()}
            chunks.append(body(**batch).joints[:, :24].cpu().numpy())
    joints = np.concatenate(chunks).astype(np.float32)
    if joints.shape != (length, 24, 3) or not np.isfinite(joints).all():
        raise RuntimeError(f"Invalid SMPL-X joints: {joints.shape}")
    return joints


def ensure_video_fps_probe() -> None:
    """Use OpenCV's FPS when Windows has no external ffprobe executable.

    The port otherwise catches ffprobe's FileNotFoundError and assumes 30 FPS,
    silently skipping the required 25→30 resample for this dance video.
    """
    if shutil.which("ffprobe") is not None:
        return
    import cv2
    import gvhmr.cli.demo as demo

    def opencv_fps(path: Path) -> float:
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"Could not determine video FPS: {path}")
        return fps

    demo.get_video_fps = opencv_fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("dance.mp4"))
    parser.add_argument("--repo", type=Path, default=Path("GVHMR-portable"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dance_gvhmr"))
    parser.add_argument("--flip-test", action="store_true",
                        help="Average mirrored image features (slower, benchmark quality setting)")
    parser.add_argument("--fast-preprocess", action="store_true",
                        help="Use GVHMR's tested fast recipe: one ViTPose pass and ByteTrack")
    parser.add_argument("--preflight", action="store_true",
                        help="Check body model and checkpoints without importing GVHMR")
    args = parser.parse_args()
    video = args.video.resolve()
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    if not (repo / "gvhmr/inference/pipelines.py").is_file():
        parser.error(f"GVHMR port checkout not found: {repo}")
    if not video.is_file() and not args.preflight:
        parser.error(f"Video not found: {video}")
    missing = missing_assets(repo)
    if missing:
        print("GVHMR assets missing:")
        for item in missing:
            print(f"  - {item}")
        if any(item.startswith("licensed SMPL-X") for item in missing):
            print("Obtain SMPL-X from https://smpl-x.is.tue.mpg.de/ .")
        if any(item.startswith("checkpoint:") for item in missing):
            print("Download public weights using the GVHMR project's checkpoint downloader.")
        raise SystemExit(2)
    if args.preflight:
        print("GVHMR assets are present")
        return
    if args.fast_preprocess and not (repo / "gvhmr/configs/recipe/fast.yaml").is_file():
        parser.error("This GVHMR checkout has no fast preprocessing recipe")

    # GVHMR's preprocessing uses YOLO, ViTPose, and HMR2 in sequence. Keep all
    # of them on CPU for this machine; the 4 GB GPU remains unused. The resident
    # model cache is disabled so each completed preprocessing model is released.
    os.environ["GVHMR_DEVICE"] = "cpu"
    os.environ["GVHMR_PREDICT_DEVICE"] = "cpu"
    os.environ["GVHMR_NO_MODEL_CACHE"] = "1"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import gvhmr  # type: ignore[import-not-found]
    ensure_video_fps_probe()

    output.mkdir(parents=True, exist_ok=True)
    staged_video = output / "internal/0_input_video.mp4"
    if staged_video.exists():
        # A prior run without ffprobe may have cached a 25 FPS copy as 30 FPS.
        import cv2
        cap = cv2.VideoCapture(str(staged_video))
        try:
            staged_fps = float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        if abs(staged_fps - 30.0) > 1.5:
            staged_video.unlink()
    pipe = gvhmr.pipeline("human-motion-recovery", device="cpu")
    result = pipe(video, static_camera=True, render=False,
                  flip_test=args.flip_test,
                  recipe="fast" if args.fast_preprocess else None,
                  output_dir=output / "internal")
    params_path = result.save_npz(output / "gvhmr_params.npz")
    # The port's convenient `joints_camera_smplx` accessor also initializes an
    # SMPL-topology conversion, which requires a *separate* licensed SMPL file.
    # We only need the registered SMPL-X model: evaluate its body joints in
    # small CPU batches, avoiding the unused SMPL conversion and a large mesh
    # allocation for the whole clip at once.
    joints = smplx_camera_joints(result.smpl_params_camera)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise RuntimeError(f"Unexpected GVHMR joint shape: {joints.shape}")
    joints_path = output / "smplx24_camera.npy"
    np.save(joints_path, joints)
    joints_path.with_suffix(".json").write_text(json.dumps({
        "fps": float(result.fps),
        "joint_order": "SMPL-X first 24 body joints",
        "units": "meters",
        "coordinates": "camera: x right, y down, z forward",
        "pose_mode": "gvhmr",
        "source": "GVHMR camera-space SMPL-X body",
    }, indent=2), encoding="utf-8")
    skeleton_path = save_sequence(joints, output / "3d_points_gvhmr.npy", result.fps)
    print(f"GVHMR finished: {len(joints)} frames at {result.fps:g} FPS")
    print(f"Raw parameters: {params_path}")
    print(f"Camera joints: {joints_path}")
    print(f"Blender-ready points: {skeleton_path}")


if __name__ == "__main__":
    main()

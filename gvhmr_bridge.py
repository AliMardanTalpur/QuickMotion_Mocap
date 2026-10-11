"""Convert GVHMR's camera-space SMPL-X joints to this project's H36M-17 rig.

GVHMR produces a 24-joint body model. The first 22 joint indices follow the
SMPL body convention; this adapter uses only those named body joints. It does
not synthesize depth or fit a pose, so the same motion can be reviewed before
and after Blender export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Target: pelvis, right leg, left leg, spine, thorax, neck, head, left arm,
# right arm. Source: SMPL/SMPL-X first 22 body joints. The full index mapping
# is documented in GVHMR-portable/gvhmr/utils/vis/skeleton.py.
SMPL24_TO_H36M17 = (
    0, 2, 5, 8, 1, 4, 7, 6, 9, 12, 15, 16, 18, 20, 17, 19, 21,
)
H36M17_NAMES = (
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)


def convert_joints(camera_joints: np.ndarray) -> np.ndarray:
    """Return metric (frames,17,3) camera XYZ with the performer near origin."""
    source = np.asarray(camera_joints, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (24, 3) or not len(source):
        raise ValueError("GVHMR camera joints must have shape (frames, 24, 3)")
    if not np.isfinite(source).all():
        raise ValueError("GVHMR camera joints contain non-finite coordinates")
    out = source[:, SMPL24_TO_H36M17].copy()
    # GVHMR camera convention: X right, Y down, Z forward. Stage 3 expects the
    # same axes. Rebase horizontal/depth origin so the rig opens near Blender's
    # world origin, while retaining frame-to-frame root translation.
    out[:, :, 0] -= float(np.median(out[:, 0, 0]))
    out[:, :, 2] -= float(np.median(out[:, 0, 2]))
    return out


def save_sequence(camera_joints: np.ndarray, output: Path, fps: float,
                  source: str = "GVHMR") -> Path:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    points = convert_joints(camera_joints)
    np.save(output, points)
    output.with_suffix(".json").write_text(json.dumps({
        "fps": float(fps),
        "joint_order": H36M17_NAMES,
        "units": "meters",
        "coordinates": "camera: x right, y down, z forward",
        "pose_mode": "gvhmr",
        "source": source,
        "source_joint_order": "SMPL-X first 24 body joints",
        "source_indices": SMPL24_TO_H36M17,
        "camera_depth_rebased": True,
    }, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joints", type=Path, required=True,
                        help="Camera-space (frames,24,3) joints from GVHMR")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joints = np.load(args.joints, allow_pickle=False)
    print(f"Saved {save_sequence(joints, args.output, args.fps)}")


if __name__ == "__main__":
    main()

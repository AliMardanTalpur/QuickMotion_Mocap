"""Create a video contact sheet comparing GVHMR 3D and detected 2D joints.

Run after gvhmr_infer.py. Green is projected 3D, yellow is measured 2D, and
red links show reprojection disagreement. The side-view 3D contact sheet from
motion_preview.py should be checked alongside this one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# COCO body keypoint -> SMPL-X body joint. The facial COCO joints are omitted:
# the source model's head joint is not the same location as a visible nose.
COCO_TO_SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21,
                 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def project(joints: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Project camera XYZ (X right, Y down, Z forward) into source pixels."""
    if joints.shape != (24, 3) or intrinsic.shape != (3, 3):
        raise ValueError("Expected 24 camera joints and a 3x3 camera matrix")
    depth = joints[:, 2]
    homogeneous = (intrinsic @ joints.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    pixels[depth <= 0] = np.nan
    return pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/dance_gvhmr"))
    parser.add_argument("--joints", type=Path, default=None,
                        help="Optional alternative 24-joint sequence to compare")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    run = args.run_dir
    params_path = run / "gvhmr_params.npz"
    video_path = run / "internal/0_input_video.mp4"
    joints = np.load(args.joints or run / "smplx24_camera.npy", allow_pickle=False)
    with np.load(params_path, allow_pickle=False) as params:
        intrinsics = params["intrinsics"]
        detections = params["kp2d"]
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"Unexpected SMPL-X joints: {joints.shape}")
    if detections.shape[:2] != (len(joints), 17):
        raise ValueError(f"Unexpected 2D detections: {detections.shape}")
    if intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(intrinsics, (len(joints), 3, 3))
    if intrinsics.shape != (len(joints), 3, 3):
        raise ValueError(f"Unexpected intrinsics: {intrinsics.shape}")
    indices = np.linspace(0, len(joints) - 1, args.samples).round().astype(int)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open GVHMR's staged video: {video_path}")
    tile_w, tile_h = 480, 270
    cols = min(4, args.samples)
    rows = (args.samples + cols - 1) // cols
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 235, dtype=np.uint8)
    residuals = []
    frame_reports = []
    try:
        for sample, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read staged video frame {frame_index + 1}")
            h, w = frame.shape[:2]
            tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            scale = np.array([tile_w / w, tile_h / h])
            measured = detections[frame_index]
            projected = project(joints[frame_index], intrinsics[frame_index])
            matched = {}
            errors = []
            for coco, smplx in COCO_TO_SMPLX.items():
                det = measured[coco]
                pred = projected[smplx]
                if np.isfinite(det[:2]).all() and float(det[2]) >= 0.25:
                    q = tuple(np.rint(det[:2] * scale).astype(int))
                    cv2.circle(tile, q, 4, (0, 220, 255), -1, cv2.LINE_AA)
                    if np.isfinite(pred).all():
                        p = tuple(np.rint(pred * scale).astype(int))
                        cv2.line(tile, q, p, (25, 25, 220), 1, cv2.LINE_AA)
                        error = float(np.linalg.norm(pred - det[:2]))
                        errors.append(error)
                        residuals.append(error)
                if np.isfinite(pred).all():
                    matched[coco] = tuple(np.rint(pred * scale).astype(int))
            for first, second in EDGES:
                if first in matched and second in matched:
                    cv2.line(tile, matched[first], matched[second],
                             (25, 220, 25), 2, cv2.LINE_AA)
            for point in matched.values():
                cv2.circle(tile, point, 3, (25, 220, 25), -1, cv2.LINE_AA)
            median_error = float(np.median(errors)) if errors else None
            label = f"frame {frame_index + 1}   median {median_error:.1f}px" if errors else f"frame {frame_index + 1}   no reliable 2D"
            cv2.rectangle(tile, (0, 0), (tile_w, 28), (25, 25, 25), -1)
            cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            row, col = divmod(sample, cols)
            sheet[row * tile_h:(row + 1) * tile_h,
                  col * tile_w:(col + 1) * tile_w] = tile
            frame_reports.append({"frame": int(frame_index + 1),
                                  "median_reprojection_error_px": median_error,
                                  "matched_joints": len(errors)})
    finally:
        capture.release()
    output = args.output or run / "review_overlay.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Cannot save {output}")
    report = {"median_reprojection_error_px": float(np.median(residuals)) if residuals else None,
              "p95_reprojection_error_px": float(np.percentile(residuals, 95)) if residuals else None,
              "samples": frame_reports,
              "legend": {"green": "projected GVHMR 3D", "yellow": "detected 2D", "red": "disagreement"}}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output}; median reprojection error: {report['median_reprojection_error_px']}")


if __name__ == "__main__":
    main()

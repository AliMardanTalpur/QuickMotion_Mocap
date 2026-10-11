"""Correct large GVHMR limb outliers with detected 2D joints and fixed lengths.

The torso, hips, shoulders, and GVHMR camera trajectory remain unchanged. A
small least-squares fit adjusts elbows/wrists as well as knees/ankles when
their projections disagree with high-confidence ViTPose observations. It
preserves limb lengths and penalizes departures from the original 3D depth.
A short temporal pass smooths the corrections.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares


LEGS = ((1, 4, 7, 10, 13, 15), (2, 5, 8, 11, 14, 16))
ARMS = ((16, 18, 20, 7, 9), (17, 19, 21, 8, 10))


def project(point: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    homogeneous = intrinsic @ point
    return homogeneous[:2] / homogeneous[2]


def fit_leg(
    original: np.ndarray, anchor: np.ndarray, prior: np.ndarray,
    observed: np.ndarray, intrinsic: np.ndarray, lengths: tuple[float, float],
    strength: float, prior_scale: float,
) -> np.ndarray:
    """Fit knee/ankle XYZ while keeping their hip and two segment lengths."""
    if strength <= 0:
        return original.copy()
    knee_conf = np.clip(float(observed[0, 2]), 0.0, 1.0)
    ankle_conf = np.clip(float(observed[1, 2]), 0.0, 1.0)
    weights = strength * np.sqrt(np.array([knee_conf, ankle_conf]))
    if max(weights) < 0.2:
        return original.copy()

    def residual(flat: np.ndarray) -> np.ndarray:
        knee, ankle = flat.reshape(2, 3)
        return np.concatenate((
            weights[0] * (project(knee, intrinsic) - observed[0, :2]) / 22.0,
            weights[1] * (project(ankle, intrinsic) - observed[1, :2]) / 22.0,
            np.array([(np.linalg.norm(knee - anchor) - lengths[0]) / 0.012,
                      (np.linalg.norm(ankle - knee) - lengths[1]) / 0.012]),
            ((knee - prior[0]) / (prior_scale * 0.80)),
            ((ankle - prior[1]) / prior_scale),
            np.array([(knee[2] - original[0, 2]) / 0.45,
                      (ankle[2] - original[1, 2]) / 0.55]),
        ))

    result = least_squares(residual, prior.ravel(), max_nfev=40,
                           bounds=([-np.inf, -np.inf, 0.5] * 2,
                                   [np.inf, np.inf, 50.0] * 2))
    return result.x.reshape(2, 3)


def refine(points: np.ndarray, keypoints: np.ndarray,
           intrinsics: np.ndarray, smoothing: float = 0.7,
           final_smoothing: float = 1.0,
           refine_arms: bool = True) -> tuple[np.ndarray, dict]:
    if points.ndim != 3 or points.shape[1:] != (24, 3):
        raise ValueError("GVHMR joints must have shape (frames, 24, 3)")
    n = len(points)
    if keypoints.shape != (n, 17, 3) or intrinsics.shape != (n, 3, 3):
        raise ValueError("2D keypoints/intrinsics must match the 24-joint sequence")
    if not (np.isfinite(points).all() and np.isfinite(keypoints).all()
            and np.isfinite(intrinsics).all()):
        raise ValueError("Input contains non-finite values")
    output = points.astype(np.float64, copy=True)
    leg_reports = []
    for hip, knee, ankle, foot, knee2d, ankle2d in LEGS:
        lengths = (float(np.median(np.linalg.norm(points[:, knee] - points[:, hip], axis=1))),
                   float(np.median(np.linalg.norm(points[:, ankle] - points[:, knee], axis=1))))
        original = points[:, [knee, ankle]].astype(np.float64)
        observation = keypoints[:, [knee2d, ankle2d]].astype(np.float64)
        strength = np.zeros(n)
        before = np.zeros((n, 2))
        for frame in range(n):
            for joint, joint2d, column in ((knee, knee2d, 0), (ankle, ankle2d, 1)):
                before[frame, column] = np.linalg.norm(
                    project(points[frame, joint], intrinsics[frame])
                    - keypoints[frame, joint2d, :2]
                )
            confident = observation[frame, :, 2] >= 0.35
            max_error = float(np.max(before[frame, confident])) if np.any(confident) else 0.0
            strength[frame] = float(np.clip((max_error - 35.0) / 90.0, 0.0, 1.0))
        first = np.stack([
            fit_leg(original[f], points[f, hip], original[f], observation[f],
                    intrinsics[f], lengths, strength[f], prior_scale=0.42)
            for f in range(n)
        ])
        delta = first - original
        if smoothing > 0:
            delta = gaussian_filter1d(delta, sigma=smoothing, axis=0, mode="nearest")
        smooth_prior = original + delta
        final = np.stack([
            fit_leg(original[f], points[f, hip], smooth_prior[f], observation[f],
                    intrinsics[f], lengths, strength[f], prior_scale=0.24)
            for f in range(n)
        ])
        output[:, knee] = final[:, 0]
        output[:, ankle] = final[:, 1]
        # SMPL-X has a measured ankle-to-foot joint. Keep GVHMR's toe direction
        # and translate that terminal point with its corrected ankle.
        output[:, foot] = points[:, foot] + (final[:, 1] - original[:, 1])
        after = np.stack([
            [np.linalg.norm(project(output[f, joint], intrinsics[f])
                            - keypoints[f, joint2d, :2])
             for joint, joint2d in ((knee, knee2d), (ankle, ankle2d))]
            for f in range(n)
        ])
        active = strength > 0.15
        leg_reports.append({
            "side": "left" if hip == 1 else "right",
            "active_frames": int(active.sum()),
            "knee_median_error_before_px": float(np.median(before[:, 0])),
            "knee_median_error_after_px": float(np.median(after[:, 0])),
            "ankle_p95_error_before_px": float(np.percentile(before[:, 1], 95)),
            "ankle_p95_error_after_px": float(np.percentile(after[:, 1], 95)),
            "max_ankle_correction_m": float(np.max(np.linalg.norm(final[:, 1] - original[:, 1], axis=1))),
            "thigh_length_p95_error_m": float(np.percentile(
                np.abs(np.linalg.norm(final[:, 0] - points[:, hip], axis=1) - lengths[0]), 95)),
            "shin_length_p95_error_m": float(np.percentile(
                np.abs(np.linalg.norm(final[:, 1] - final[:, 0], axis=1) - lengths[1]), 95)),
        })
    # Fitting each frame can follow one-frame 2D detector noise too closely.
    # Smooth only the *correction*, keeping GVHMR's original fast motion, then
    # re-enforce both bone lengths exactly along the smoothed directions.
    if final_smoothing > 0:
        for hip, knee, ankle, foot, _, _ in LEGS:
            before_ankle = output[:, ankle].copy()
            source_pair = points[:, [knee, ankle]]
            correction = output[:, [knee, ankle]] - source_pair
            correction = gaussian_filter1d(correction, sigma=final_smoothing,
                                           axis=0, mode="nearest")
            output[:, [knee, ankle]] = source_pair + correction
            thigh = float(np.median(np.linalg.norm(points[:, knee] - points[:, hip], axis=1)))
            shin = float(np.median(np.linalg.norm(points[:, ankle] - points[:, knee], axis=1)))
            vector = output[:, knee] - output[:, hip]
            output[:, knee] = output[:, hip] + thigh * vector / np.maximum(
                np.linalg.norm(vector, axis=1)[:, None], 1e-8)
            vector = output[:, ankle] - output[:, knee]
            output[:, ankle] = output[:, knee] + shin * vector / np.maximum(
                np.linalg.norm(vector, axis=1)[:, None], 1e-8)
            output[:, foot] += output[:, ankle] - before_ankle
    for report, (hip, knee, ankle, _, knee2d, ankle2d) in zip(leg_reports, LEGS):
        after = np.array([
            [np.linalg.norm(project(output[f, joint], intrinsics[f])
                            - keypoints[f, joint2d, :2])
             for joint, joint2d in ((knee, knee2d), (ankle, ankle2d))]
            for f in range(n)
        ])
        report["knee_median_error_after_px"] = float(np.median(after[:, 0]))
        report["ankle_p95_error_after_px"] = float(np.percentile(after[:, 1], 95))
        report["max_ankle_correction_m"] = float(np.max(
            np.linalg.norm(output[:, ankle] - points[:, ankle], axis=1)))
        thigh = float(np.median(np.linalg.norm(points[:, knee] - points[:, hip], axis=1)))
        shin = float(np.median(np.linalg.norm(points[:, ankle] - points[:, knee], axis=1)))
        report["thigh_length_p95_error_m"] = float(np.percentile(
            np.abs(np.linalg.norm(output[:, knee] - points[:, hip], axis=1) - thigh), 95))
        report["shin_length_p95_error_m"] = float(np.percentile(
            np.abs(np.linalg.norm(output[:, ankle] - output[:, knee], axis=1) - shin), 95))
    arm_reports = []
    if refine_arms:
        for shoulder, elbow, wrist, elbow2d, wrist2d in ARMS:
            lengths = (
                float(np.median(np.linalg.norm(points[:, elbow] - points[:, shoulder], axis=1))),
                float(np.median(np.linalg.norm(points[:, wrist] - points[:, elbow], axis=1))),
            )
            original = points[:, [elbow, wrist]].astype(np.float64)
            observations = keypoints[:, [elbow2d, wrist2d]].astype(np.float64)
            before = np.array([
                [np.linalg.norm(project(points[f, joint], intrinsics[f])
                                - keypoints[f, joint2d, :2])
                 for joint, joint2d in ((elbow, elbow2d), (wrist, wrist2d))]
                for f in range(n)
            ])
            confidence = observations[:, :, 2] >= 0.5
            max_error = np.max(np.where(confidence, before, 0.0), axis=1)
            strength = np.clip((max_error - 25.0) / 90.0, 0.0, 1.0)
            first = np.stack([
                fit_leg(original[f], points[f, shoulder], original[f], observations[f],
                        intrinsics[f], lengths, strength[f], prior_scale=0.35)
                for f in range(n)
            ])
            correction = first - original
            if smoothing > 0:
                correction = gaussian_filter1d(correction, sigma=smoothing,
                                               axis=0, mode="nearest")
            smooth_prior = original + correction
            final = np.stack([
                fit_leg(original[f], points[f, shoulder], smooth_prior[f], observations[f],
                        intrinsics[f], lengths, strength[f], prior_scale=0.22)
                for f in range(n)
            ])
            correction = final - original
            if final_smoothing > 0:
                correction = gaussian_filter1d(correction, sigma=final_smoothing,
                                               axis=0, mode="nearest")
            final = original + correction
            # Smoothing the correction may slightly change limb lengths.
            # Re-project each segment onto its fixed-length sphere.
            upper = final[:, 0] - output[:, shoulder]
            final[:, 0] = output[:, shoulder] + lengths[0] * upper / np.maximum(
                np.linalg.norm(upper, axis=1)[:, None], 1e-8)
            lower = final[:, 1] - final[:, 0]
            final[:, 1] = final[:, 0] + lengths[1] * lower / np.maximum(
                np.linalg.norm(lower, axis=1)[:, None], 1e-8)
            output[:, [elbow, wrist]] = final
            after = np.array([
                [np.linalg.norm(project(output[f, joint], intrinsics[f])
                                - keypoints[f, joint2d, :2])
                 for joint, joint2d in ((elbow, elbow2d), (wrist, wrist2d))]
                for f in range(n)
            ])
            active = strength > 0.15
            arm_reports.append({
                "side": "left" if shoulder == 16 else "right",
                "active_frames": int(active.sum()),
                "elbow_median_error_before_px": float(np.median(before[:, 0])),
                "elbow_median_error_after_px": float(np.median(after[:, 0])),
                "wrist_p95_error_before_px": float(np.percentile(before[:, 1], 95)),
                "wrist_p95_error_after_px": float(np.percentile(after[:, 1], 95)),
                "max_wrist_correction_m": float(np.max(np.linalg.norm(
                    final[:, 1] - original[:, 1], axis=1))),
                "upper_arm_length_p95_error_m": float(np.percentile(np.abs(
                    np.linalg.norm(final[:, 0] - output[:, shoulder], axis=1)
                    - lengths[0]), 95)),
                "forearm_length_p95_error_m": float(np.percentile(np.abs(
                    np.linalg.norm(final[:, 1] - final[:, 0], axis=1)
                    - lengths[1]), 95)),
            })
    return output.astype(np.float32), {
        "method": "2D-guided GVHMR limb refinement",
        "smoothing_sigma_frames": smoothing,
        "final_smoothing_sigma_frames": final_smoothing,
        "legs": leg_reports,
        "arms": arm_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/dance_gvhmr"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoothing", type=float, default=0.7)
    parser.add_argument("--final-smoothing", type=float, default=1.0)
    parser.add_argument("--arms", choices=("on", "off"), default="on",
                        help="Fit high-confidence 2D elbows and wrists")
    args = parser.parse_args()
    if (not np.isfinite(args.smoothing) or args.smoothing < 0
            or not np.isfinite(args.final_smoothing) or args.final_smoothing < 0):
        parser.error("Smoothing values must be nonnegative")
    run = args.run_dir
    points = np.load(run / "smplx24_camera.npy", allow_pickle=False)
    with np.load(run / "gvhmr_params.npz", allow_pickle=False) as params:
        keypoints = params["kp2d"]
        intrinsics = params["intrinsics"]
        fps = float(params["fps"])
    refined, report = refine(points, keypoints, intrinsics,
                             args.smoothing, args.final_smoothing,
                             refine_arms=args.arms == "on")
    report.update({"fps": fps, "joint_order": "SMPL-X first 24 body joints",
                   "units": "meters", "coordinates": "camera: x right, y down, z forward",
                   "pose_mode": "gvhmr-refined"})
    output = args.output or run / "smplx24_refined.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, refined)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    for limb in report["legs"] + report["arms"]:
        print(limb)


if __name__ == "__main__":
    main()

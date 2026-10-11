"""CPU-only keypoint and skeleton conditioning for monocular motion capture."""

from __future__ import annotations

import numpy as np


# Human3.6M-17 kinematic tree. The spatial regularizer keeps one median length
# per edge, preventing frame-to-frame bone stretching from the lifting network.
H36M_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)


def gaussian_time(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(values) < 2:
        return values.copy()
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    padded = np.pad(values, ((radius, radius),) + ((0, 0),) * (values.ndim - 1), mode="edge")
    result = np.zeros_like(values, dtype=np.float32)
    for index, weight in enumerate(weights):
        result += np.float32(weight) * padded[index:index + len(values)]
    return result


def clean_2d(xy: np.ndarray, confidence: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    """Repair isolated bad 2D locations and lightly smooth detector noise."""
    result = xy.astype(np.float32).copy()
    if len(result) > 2:
        midpoint = (result[:-2] + result[2:]) / 2
        deviation = np.linalg.norm(result[1:-1] - midpoint, axis=-1)
        motion = np.linalg.norm(result[2:] - result[:-2], axis=-1)
        # Only an isolated discontinuity, especially a low-confidence one, is
        # replaced. Fast coherent punches remain intact.
        isolated = (deviation > np.maximum(20.0, 1.4 * motion)) & (
            confidence[1:-1] < 0.65
        )
        result[1:-1] = np.where(isolated[..., None], midpoint, result[1:-1])
    return gaussian_time(result, sigma)


def clean_3d(points: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Remove isolated 3D flips, smooth jitter, and enforce stable bone lengths."""
    result = points.astype(np.float32).copy()
    if len(result) > 2:
        midpoint = (result[:-2] + result[2:]) / 2
        deviation = np.linalg.norm(result[1:-1] - midpoint, axis=-1)
        motion = np.linalg.norm(result[2:] - result[:-2], axis=-1)
        isolated = (deviation > 0.10) & (deviation > 1.8 * motion)
        result[1:-1] = np.where(isolated[..., None], midpoint, result[1:-1])
    result = gaussian_time(result, sigma)
    return stabilize_lengths(result)


def stabilize_lengths(points: np.ndarray) -> np.ndarray:
    """Project joints onto a skeleton with one median length per bone."""
    result = points.astype(np.float32).copy()
    lengths = {}
    for joint, parent in enumerate(H36M_PARENTS):
        if parent >= 0:
            lengths[joint] = float(np.median(np.linalg.norm(result[:, joint] - result[:, parent], axis=1)))
    for joint, parent in enumerate(H36M_PARENTS):
        if parent < 0:
            continue
        direction = result[:, joint] - result[:, parent]
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction /= np.maximum(norm, 1e-6)
        result[:, joint] = result[:, parent] + direction * lengths[joint]
    result -= result[:, :1]
    return result


def regularize_front_depth(points: np.ndarray) -> np.ndarray:
    """Reduce implausible front-to-back lean for a mostly frontal camera.

    Monocular depth is uncertain. Keeping each depth offset relative to its
    parent retains turns and limb foreshortening, while constraining the torso
    more strongly than the arms. X/Y camera coordinates stay unchanged until
    the small adjustment needed to restore fixed bone lengths.
    """
    result = points.astype(np.float32).copy()
    edge_depth = np.array([
        1.0, 0.35, 0.40, 0.45, 0.35, 0.40, 0.45,
        0.20, 0.20, 0.25, 0.25,
        0.35, 0.65, 0.70, 0.35, 0.65, 0.70,
    ], dtype=np.float32)
    for joint, parent in enumerate(H36M_PARENTS):
        if parent >= 0:
            result[:, joint, 2] = (
                result[:, parent, 2]
                + edge_depth[joint] * (points[:, joint, 2] - points[:, parent, 2])
            )
    return stabilize_lengths(result)


def reduce_footskate(
    points: np.ndarray, fps: float,
    max_correction: float = 0.10,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Gently correct world root XY during short, likely foot contacts.

    `points` uses Blender XYZ with the floor at Z=0. Every joint receives the
    same translation, so this cannot distort the inferred body pose. Contact
    segments are short and require both a low ankle and slow ankle motion;
    large real steps therefore keep their original root trajectory.
    """
    result = np.asarray(points, dtype=np.float32).copy()
    frames = len(result)
    if frames < 6:
        return result, {"foot_contact_segments": 0, "foot_contact_frames": 0}
    contact_weight = np.zeros(frames, dtype=np.float64)
    target = np.zeros((frames, 2), dtype=np.float64)
    original_drift = []
    contacts = []
    for ankle in (3, 6):
        foot = result[:, ankle]
        speed = np.linalg.norm(np.diff(foot[:, :2], axis=0), axis=1) * fps
        speed = np.r_[speed[0], (speed[:-1] + speed[1:]) / 2, speed[-1]]
        candidate = (foot[:, 2] < 0.045) & (speed < 0.30)
        start = 0
        while start < frames:
            if not candidate[start]:
                start += 1
                continue
            end = start + 1
            while end < frames and candidate[end] and end - start < 24:
                if np.linalg.norm(foot[end, :2] - foot[start, :2]) > 0.10:
                    break
                end += 1
            if end - start >= 5:
                anchor = np.median(foot[start:end, :2], axis=0)
                contacts.append((ankle, start, end, anchor))
                contact_weight[start:end] += 20.0
                target[start:end] += 20.0 * (anchor - foot[start:end, :2])
                original_drift.extend(np.linalg.norm(foot[start:end, :2] - anchor, axis=1))
            start = end

    if not contacts:
        return result, {"foot_contact_segments": 0, "foot_contact_frames": 0}

    # Solve a tridiagonal least-squares system: small correction, smooth
    # correction, and agreement with each planted ankle's fixed anchor.
    smooth = 60.0
    diagonal = 2.0 + contact_weight + 2.0 * smooth
    diagonal[0] -= smooth
    diagonal[-1] -= smooth
    upper = np.full(frames - 1, -smooth, dtype=np.float64)
    rhs = target.copy()
    for i in range(1, frames):
        factor = -smooth / diagonal[i - 1]
        diagonal[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]
    correction = np.zeros_like(rhs)
    correction[-1] = rhs[-1] / diagonal[-1]
    for i in range(frames - 2, -1, -1):
        correction[i] = (rhs[i] - upper[i] * correction[i + 1]) / diagonal[i]
    length = np.linalg.norm(correction, axis=1)
    correction *= np.minimum(1.0, max_correction / np.maximum(length, 1e-9))[:, None]
    result[:, :, :2] += correction[:, None, :]
    corrected_drift = []
    for ankle, start, end, anchor in contacts:
        corrected_drift.extend(np.linalg.norm(result[start:end, ankle, :2] - anchor, axis=1))
    return result, {
        "foot_contact_segments": len(contacts),
        "foot_contact_frames": int(np.count_nonzero(contact_weight)),
        "foot_contact_drift_before_m": float(np.mean(original_drift)),
        "foot_contact_drift_after_m": float(np.mean(corrected_drift)),
        "root_correction_max_m": float(np.max(np.linalg.norm(correction, axis=1))),
    }

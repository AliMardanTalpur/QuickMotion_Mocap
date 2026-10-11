"""2D-guided 3D reconstruction for poses outside VideoPose3D's training set."""

from __future__ import annotations

import numpy as np

from motion_processing import H36M_PARENTS, gaussian_time


COCO_MATCHES = ((1, 12), (2, 14), (3, 16), (4, 11), (5, 13), (6, 15),
                (11, 5), (12, 7), (13, 9), (14, 6), (15, 8), (16, 10))


def model_reprojection_error(model: np.ndarray, keypoints: np.ndarray) -> float:
    """Median joint error divided by visible body size after a 2D similarity fit."""
    errors = []
    for frame in range(len(model)):
        joints = np.array([model[frame, j, :2] for j, _ in COCO_MATCHES])
        pixels = np.array([keypoints[frame, c, :2] for _, c in COCO_MATCHES])
        confidence = np.array([keypoints[frame, c, 2] for _, c in COCO_MATCHES])
        valid = (confidence > 0.3) & np.isfinite(pixels).all(axis=1)
        if valid.sum() < 6:
            continue
        joints, pixels = joints[valid], pixels[valid]
        centered = joints - joints.mean(axis=0)
        scale = np.sum(centered * (pixels - pixels.mean(axis=0))) / max(
            np.sum(centered ** 2), 1e-8
        )
        residual = np.linalg.norm(scale * centered + pixels.mean(axis=0) - pixels, axis=1)
        body_size = np.linalg.norm(keypoints[frame, 5, :2] - keypoints[frame, 15, :2])
        if not np.isfinite(body_size):
            body_size = np.ptp(pixels[:, 1])
        errors.append(float(np.median(residual) / max(body_size, 100.0)))
    return float(np.median(errors)) if errors else float("inf")


def coco_to_h36m_screen(xy: np.ndarray) -> np.ndarray:
    """Map observed COCO joints and inferred torso midpoints to H36M order."""
    out = np.empty((len(xy), 17, 2), dtype=np.float32)
    pelvis = (xy[:, 11] + xy[:, 12]) / 2
    thorax = (xy[:, 5] + xy[:, 6]) / 2
    out[:, 0] = pelvis
    out[:, 1:4] = xy[:, (12, 14, 16)]
    out[:, 4:7] = xy[:, (11, 13, 15)]
    out[:, 7] = pelvis * 0.55 + thorax * 0.45
    out[:, 8] = thorax
    out[:, 9] = thorax * 0.75 + xy[:, 0] * 0.25
    out[:, 10] = xy[:, 0]
    out[:, 11:14] = xy[:, (5, 7, 9)]
    out[:, 14:17] = xy[:, (6, 8, 10)]
    return out


def _smooth_depth_signs(
    magnitude: np.ndarray, predicted: np.ndarray,
    preferred: np.ndarray | None = None,
    preference_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Pick front/back edge signs with temporal continuity."""
    n = len(magnitude)
    choices = np.stack((-magnitude, magnitude), axis=1)
    predicted = gaussian_time(predicted.astype(np.float32), 1.5)
    cost = np.empty((n, 2), dtype=np.float64)
    previous = np.zeros((n, 2), dtype=np.int8)
    unary = 0.35 * (choices - predicted[:, None]) ** 2
    if preferred is not None and preference_weight is not None:
        unary += preference_weight[:, None] * (choices - preferred[:, None]) ** 2
    cost[0] = unary[0]
    for t in range(1, n):
        for state in (0, 1):
            options = cost[t - 1] + 2.5 * (choices[t, state] - choices[t - 1]) ** 2
            previous[t, state] = int(np.argmin(options))
            cost[t, state] = options[previous[t, state]] + unary[t, state]
    state = int(np.argmin(cost[-1]))
    depth = np.empty(n, dtype=np.float32)
    for t in range(n - 1, -1, -1):
        depth[t] = choices[t, state]
        state = int(previous[t, state])
    return depth


def reconstruct_from_2d(
    xy: np.ndarray, model: np.ndarray, target_height: float,
    depth_prior: str = "model",
) -> tuple[np.ndarray, dict[str, float]]:
    """Preserve 2D limb geometry, infer the smallest depth needed for rigid bones."""
    screen = coco_to_h36m_screen(xy)
    factor = target_height / 1.65
    nominal = np.array([
        0, .11, .43, .42, .11, .43, .42,
        .20, .23, .11, .20, .20, .31, .27, .20, .31, .27,
    ], dtype=np.float32) * factor
    # A high percentile of observed leg lengths estimates meters per pixel
    # without assuming the acrobat is upright at any particular frame.
    leg_edges = (2, 3, 5, 6)
    ratios = []
    for joint in leg_edges:
        parent = H36M_PARENTS[joint]
        distance = np.linalg.norm(screen[:, joint] - screen[:, parent], axis=1)
        ratios.append(nominal[joint] / max(float(np.percentile(distance, 95)), 1))
    pixel_scale = float(np.median(ratios))
    local = (screen - screen[:, :1]) * pixel_scale
    result = np.zeros((len(xy), 17, 3), dtype=np.float32)
    result[:, :, :2] = local
    result[:, 0, :2] = (screen[:, 0] - screen[0, 0]) * pixel_scale
    result[:, 1:, :2] += result[:, :1, :2]

    lengths = nominal.copy()
    for joint, parent in enumerate(H36M_PARENTS):
        if parent < 0:
            continue
        projected = np.linalg.norm(screen[:, joint] - screen[:, parent], axis=1) * pixel_scale
        lengths[joint] = max(lengths[joint], float(np.percentile(projected, 97)) * 1.005)
    # Equivalent left/right bones have the same length even when one side is
    # occluded or foreshortened for much of the video.
    for left, right in ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)):
        common = max(lengths[left], lengths[right])
        lengths[left] = lengths[right] = common

    leg_px = np.median([
        np.median(np.linalg.norm(screen[:, j] - screen[:, H36M_PARENTS[j]], axis=1))
        for j in leg_edges
    ])
    ankle_y = (screen[:, 3, 1] + screen[:, 6, 1]) / 2
    upright = (
        (screen[:, 0, 1] - screen[:, 10, 1] > 0.8 * leg_px)
        & (ankle_y - screen[:, 0, 1] > 0.8 * leg_px)
    )
    upright_weight = gaussian_time(upright.astype(np.float32), 2.0) * 6.0
    clamped_frames = 0
    for joint, parent in enumerate(H36M_PARENTS):
        if parent < 0:
            continue
        vector = result[:, joint, :2] - result[:, parent, :2]
        planar = np.linalg.norm(vector, axis=1)
        too_long = planar > lengths[joint]
        clamped_frames += int(np.count_nonzero(too_long))
        vector *= np.minimum(1.0, lengths[joint] / np.maximum(planar, 1e-8))[:, None]
        result[:, joint, :2] = result[:, parent, :2] + vector
        planar = np.linalg.norm(vector, axis=1)
        magnitude = np.sqrt(np.maximum(lengths[joint] ** 2 - planar ** 2, 0.0))
        predicted = model[:, joint, 2] - model[:, parent, 2]
        preferred = None
        weight = None
        if depth_prior == "planar":
            # Keep complete limbs near the camera plane. Fixed bone lengths
            # still require some depth where a segment is foreshortened, but
            # alternate signs instead of stacking every edge in one direction.
            preferred = -result[:, parent, 2]
            weight = np.full(len(xy), 3.0, dtype=np.float32)
        if joint in (3, 6):
            # When upright, a foot usually lies near its hip in camera depth.
            hip = 1 if joint == 3 else 4
            if depth_prior != "planar":
                preferred = result[:, hip, 2] - result[:, parent, 2]
                weight = upright_weight
        elif joint in (13, 16):
            shoulder = 11 if joint == 13 else 14
            if depth_prior != "planar":
                preferred = result[:, shoulder, 2] - result[:, parent, 2]
                weight = upright_weight * 0.6
        result[:, joint, 2] = result[:, parent, 2] + _smooth_depth_signs(
            magnitude, predicted, preferred, weight
        )

    return result, {
        "image_guided_meters_per_pixel": pixel_scale,
        "image_guided_clamped_joint_frames": clamped_frames,
        "image_guided_upright_frames": int(np.count_nonzero(upright)),
        "image_guided_depth_prior": depth_prior,
    }

"""Conservative planted-foot root correction using predicted SMPL joints."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .contracts import SmplSequence


class GroundAligner:
    """Set a single estimated support plane to coordinate zero for the clip."""

    def run(self, motion: SmplSequence) -> tuple[SmplSequence, dict]:
        result = SmplSequence(
            motion.rotations.copy(), motion.betas.copy(), motion.translation.copy(),
            motion.joints.copy(), motion.valid.copy(), motion.confidence.copy(),
        )
        upright = np.nanmedian(
            motion.joints[:, 15, 1] - motion.joints[:, (7, 8), 1].mean(axis=1)
        )
        up = 1.0 if upright >= 0 else -1.0
        # Feet support standing frames; wrists support handstands. A low
        # percentile is robust to occasional bad joints or airborne frames.
        support = up * motion.joints[:, (7, 8, 20, 21), 1]
        ground = float(np.nanpercentile(np.nanmin(support, axis=1), 15))
        offset = ground / up
        result.translation[:, 1] -= offset
        result.joints[:, :, 1] -= offset
        return result, {
            "ground_axis_sign": up,
            "estimated_floor_offset_m": offset,
            "ground_alignment_is_estimated": True,
        }


class FootContactCorrector:
    def __init__(self, max_root_shift_m: float = .15) -> None:
        self.max_root_shift_m = max_root_shift_m

    def run(self, motion: SmplSequence, fps: float) -> tuple[SmplSequence, dict]:
        result = SmplSequence(
            motion.rotations.copy(), motion.betas.copy(), motion.translation.copy(),
            motion.joints.copy(), motion.valid.copy(), motion.confidence.copy(),
        )
        n = len(motion.valid)
        if n < 5:
            return result, {"contact_segments": 0, "contact_frames": 0}
        # SMPL indices: pelvis 0, ankles 7/8, wrists 20/21, head 15.
        head_above_feet = np.nanmedian(
            motion.joints[:, 15, 1] - motion.joints[:, (7, 8), 1].mean(axis=1)
        )
        up = 1.0 if head_above_feet >= 0 else -1.0
        support = up * motion.joints[:, (7, 8, 20, 21), 1]
        ground = float(np.nanpercentile(np.nanmin(support, axis=1), 15))
        correction = np.zeros((n, 2), np.float32)
        weight = np.zeros(n, np.float32)
        before = []
        anchors: list[tuple[int, int, int, np.ndarray]] = []
        segments = 0
        for ankle in (7, 8):
            foot = motion.joints[:, ankle]
            foot_height = up * foot[:, 1] - ground
            below_pelvis = up * (motion.joints[:, 0, 1] - foot[:, 1]) > .32
            speed = np.linalg.norm(np.diff(foot[:, (0, 2)], axis=0), axis=1) * fps
            speed = np.r_[speed[0], (speed[1:] + speed[:-1]) / 2, speed[-1]]
            candidate = ((foot_height > -.08) & (foot_height < .12)
                         & (speed < .55) & below_pelvis & motion.valid)
            start = 0
            while start < n:
                if not candidate[start]:
                    start += 1
                    continue
                end = start + 1
                while end < n and candidate[end] and end - start < int(2 * fps):
                    end += 1
                if end - start >= 3:
                    anchor = np.median(foot[start:end][:, (0, 2)], axis=0)
                    delta = anchor - foot[start:end][:, (0, 2)]
                    correction[start:end] += delta
                    weight[start:end] += 1
                    before.extend(np.linalg.norm(delta, axis=1).tolist())
                    anchors.append((ankle, start, end, anchor))
                    segments += 1
                start = end
        active = weight > 0
        correction[active] /= weight[active, None]
        # Smooth entry/exit so contact correction cannot pop the root.
        correction = gaussian_filter1d(correction, 1.5, axis=0, mode="nearest")
        size = np.linalg.norm(correction, axis=1)
        correction *= np.minimum(1, self.max_root_shift_m / np.maximum(size, 1e-8))[:, None]
        result.translation[:, 0] += correction[:, 0]
        result.translation[:, 2] += correction[:, 1]
        result.joints[:, :, 0] += correction[:, 0, None]
        result.joints[:, :, 2] += correction[:, 1, None]
        after = [
            float(np.linalg.norm(result.joints[i, ankle, (0, 2)] - anchor))
            for ankle, start, end, anchor in anchors for i in range(start, end)
        ]
        return result, {
            "contact_segments": segments,
            "contact_frames": int(active.sum()),
            "contact_drift_before_m": float(np.mean(before)) if before else None,
            "contact_drift_after_m": float(np.mean(after)) if after else None,
            "root_shift_max_m": float(np.max(np.linalg.norm(correction, axis=1))),
            "estimated_ground_y": ground / up,
            "ground_axis_sign": up,
            "contact_is_heuristic": True,
        }

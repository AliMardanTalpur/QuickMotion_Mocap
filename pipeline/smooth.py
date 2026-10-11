"""Rotation-aware offline smoothing of SMPL motion."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .contracts import SmplSequence


def _fill(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    result = values.copy()
    samples = np.flatnonzero(known)
    if len(samples) == 0:
        raise ValueError("No solved frames to smooth")
    timeline = np.arange(len(values))
    for column in range(values.shape[1]):
        result[:, column] = np.interp(timeline, samples, values[samples, column])
    return result


def smooth_rotations(rotations: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    n, joints = rotations.shape[:2]
    out = np.empty_like(rotations, dtype=np.float32)
    for joint in range(joints):
        quat = Rotation.from_matrix(rotations[:, joint]).as_quat()
        # q and -q encode the same rotation. Remove sign jumps before any
        # temporal filter, then renormalize onto SO(3) afterwards.
        for i in range(1, n):
            if np.dot(quat[i], quat[i - 1]) < 0:
                quat[i] *= -1
        if n >= 5:
            window = min(7, n if n % 2 else n - 1)
            quat = savgol_filter(quat, window_length=window, polyorder=2, axis=0)
        quat = gaussian_filter1d(quat, sigma=sigma, axis=0, mode="nearest")
        quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
        out[:, joint] = Rotation.from_quat(quat).as_matrix().astype(np.float32)
    return out


class MotionFilter:
    def __init__(self, sigma: float = 1.2) -> None:
        if sigma < 0:
            raise ValueError("sigma must be nonnegative")
        self.sigma = sigma

    def run(self, raw: SmplSequence) -> SmplSequence:
        available = np.isfinite(raw.rotations).all(axis=(1, 2, 3))
        n = len(raw.valid)
        rotations = np.empty((n, 24, 3, 3), np.float32)
        for joint in range(24):
            quat = np.full((n, 4), np.nan, np.float64)
            quat[available] = Rotation.from_matrix(raw.rotations[available, joint]).as_quat()
            observed = np.flatnonzero(available)
            for position in range(1, len(observed)):
                current, previous = observed[position], observed[position - 1]
                if np.dot(quat[current], quat[previous]) < 0:
                    quat[current] *= -1
            quat = _fill(quat, available)
            quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
            rotations[:, joint] = Rotation.from_quat(quat).as_matrix()
        rotations = smooth_rotations(rotations, self.sigma)
        translation = _fill(raw.translation, available)
        translation = gaussian_filter1d(translation, self.sigma, axis=0, mode="nearest")
        joints = raw.joints.copy()
        for joint in range(24):
            joints[:, joint] = _fill(joints[:, joint], available)
        return SmplSequence(rotations, raw.betas.copy(), translation.astype(np.float32),
                            joints.astype(np.float32), raw.valid.copy(), raw.confidence.copy())

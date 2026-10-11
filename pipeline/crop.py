"""Square ROI generation with reversible pixel-coordinate transforms."""

from __future__ import annotations

import cv2
import numpy as np


def square_roi(box: np.ndarray, keypoints: np.ndarray, margin: float = 0.18) -> np.ndarray:
    xy = keypoints[:, :2]
    trusted = (keypoints[:, 2] >= .35) & np.isfinite(xy).all(axis=1)
    if trusted.any():
        extent = np.stack((np.minimum(box[:2], xy[trusted].min(axis=0)),
                           np.maximum(box[2:], xy[trusted].max(axis=0))))
        low, high = extent
    else:
        low, high = box[:2], box[2:]
    center = (low + high) / 2
    side = max(float(np.max(high - low)) * (1 + 2 * margin), 64.0)
    return np.array([center[0] - side / 2, center[1] - side / 2,
                     center[0] + side / 2, center[1] + side / 2], dtype=np.float32)


def crop_square(frame: np.ndarray, roi: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    if size < 64:
        raise ValueError("Crop size must be at least 64")
    side = float(roi[2] - roi[0])
    if side <= 0:
        raise ValueError("Invalid square crop")
    crop_to_image = np.array([
        [side / size, 0, roi[0]],
        [0, side / size, roi[1]],
        [0, 0, 1],
    ], dtype=np.float32)
    image_to_crop = cv2.invertAffineTransform(crop_to_image[:2])
    crop = cv2.warpAffine(frame, image_to_crop, (size, size),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return crop, crop_to_image

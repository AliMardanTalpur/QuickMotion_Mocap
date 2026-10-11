"""Optional static-camera recovery for white shoes missed by COCO pose models.

This is deliberately opt-in: it assumes a mostly fixed camera, white shoes,
and a background that can be estimated from the video. It runs on the CPU
after the pose detector has been released from GPU memory.
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import cv2
import numpy as np


def _resize(frame: np.ndarray, width: int = 960) -> np.ndarray:
    height = round(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _shoe_candidates(frame: np.ndarray, background: np.ndarray, hip: np.ndarray,
                     scale: float) -> list[dict]:
    difference = cv2.absdiff(frame, background)
    foreground = (np.max(difference, axis=2) > 35).astype(np.uint8) * 255
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 150)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(cv2.bitwise_and(foreground, white),
                            cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        aspect = width / max(height, 1)
        center = centers[index] * scale
        if not (180 <= area <= 3000 and 0.68 <= aspect <= 4.0):
            continue
        if np.linalg.norm(center - hip) > 650:
            continue
        ys, xs = np.where(labels[y:y + height, x:x + width] == index)
        pixels = np.stack((xs + x, ys + y), axis=1).astype(np.float32) * scale
        nearest = np.argsort(np.linalg.norm(pixels - hip, axis=1))[:max(5, len(pixels) // 30)]
        ankle = np.mean(pixels[nearest], axis=0)
        candidates.append({"center": center, "ankle": ankle, "area": area})
    return candidates


def _dark_leg_endpoints(frame: np.ndarray, background: np.ndarray,
                        pose: np.ndarray, scale: float,
                        leg_pixel: float) -> list[np.ndarray]:
    """Find tips of moving dark trouser branches connected to the pelvis."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    difference = np.max(cv2.absdiff(frame, background), axis=2)
    mask = ((gray < 100) & (difference > 20)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return []
    hip = np.nanmean(pose[(11, 12), :2], axis=0) / scale
    # The torso and trousers form one large moving dark component in this clip.
    component = min(
        range(1, count),
        key=lambda index: np.linalg.norm(centers[index] - hip) - .005 * stats[index, 4],
    )
    if stats[component, 4] < 1000:
        return []
    ys, xs = np.where(labels == component)
    pixels = np.stack((xs, ys), axis=1)
    distances = np.linalg.norm(pixels - hip, axis=1)
    # A pose model may itself label a raised shoe as a wrist during acrobatics;
    # do not exclude wrist locations from image evidence.
    excluded = [pose[j, :2] / scale for j in (0, 5, 6)
                if np.isfinite(pose[j, :2]).all()]
    chosen = []
    for index in np.argsort(distances)[::-1]:
        point = pixels[index].astype(np.float32)
        if distances[index] * scale < max(150, .8 * leg_pixel):
            break
        if any(np.linalg.norm(point - other) < 60 for other in excluded):
            continue
        if any(np.linalg.norm(point - other) * scale < max(120, .65 * leg_pixel)
               for other in chosen):
            continue
        chosen.append(point)
        if len(chosen) == 2:
            break
    return [point * scale for point in chosen]


def recover_white_shoes(video_path: Path, keypoints: np.ndarray) -> tuple[np.ndarray, dict]:
    """Repair collapsed ankle tracks using foreground white-shoe components."""
    result = keypoints.copy()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen {video_path} for shoe recovery")
    frames = len(result)
    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = original_width / 960.0
    samples = []
    for index in np.linspace(0, frames - 1, min(frames, 31)).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            samples.append(_resize(frame))
    if len(samples) < 5:
        capture.release()
        raise RuntimeError("Too few video frames to estimate a static background")
    background = np.median(np.stack(samples), axis=0).astype(np.uint8)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    hip = np.nanmean(result[:, (11, 12), :2], axis=1)
    leg_sizes = [
        np.linalg.norm(result[:, a, :2] - result[:, b, :2], axis=1)
        for a, b in ((11, 13), (13, 15), (12, 14), (14, 16))
    ]
    leg_pixel = float(np.nanmedian(np.stack(leg_sizes)))
    ankle_mean_y = np.nanmean(result[:, (15, 16), 1], axis=1)
    upright = (
        (hip[:, 1] - result[:, 0, 1] > .8 * leg_pixel)
        & (ankle_mean_y - hip[:, 1] > .9 * leg_pixel)
        & (np.min(result[:, (15, 16), 2], axis=1) > .5)
    )
    duplicate = np.linalg.norm(result[:, 15, :2] - result[:, 16, :2], axis=1) < max(
        70, 0.55 * leg_pixel
    )
    proposals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    try:
        for frame_index in range(frames):
            ok, frame = capture.read()
            if not ok:
                break
            if not np.isfinite(hip[frame_index]).all():
                continue
            small = _resize(frame)
            candidates = _shoe_candidates(small, background, hip[frame_index], scale)
            high_shoe = any(
                c["ankle"][1] < hip[frame_index, 1] - leg_pixel
                and c["area"] > 250
                and np.linalg.norm(c["ankle"] - hip[frame_index]) > 1.2 * leg_pixel
                for c in candidates
            )
            dark = _dark_leg_endpoints(small, background, result[frame_index], scale, leg_pixel)
            # A floor pose can look "upright" to YOLO when it assigns both
            # ankles to the grounded shoe. A high silhouette tip is stronger
            # evidence for the raised leg even if its white shoe is occluded.
            high_dark = any(
                point[1] < hip[frame_index, 1] - 1.1 * leg_pixel
                and np.linalg.norm(point - hip[frame_index]) > 1.5 * leg_pixel
                for point in dark
            )
            if upright[frame_index] and not (high_shoe or (duplicate[frame_index] and high_dark)):
                duplicate[frame_index] = False
                continue
            # If the detector already has two distinct ankles at both visible
            # silhouette tips, white patches on a hand or shirt must not move
            # either foot. This is especially important after the floor work.
            original_ankles = result[frame_index, (15, 16), :2]
            if len(dark) == 2 and min(point[1] for point in dark) > hip[frame_index, 1] + .4 * leg_pixel:
                direct = sum(np.linalg.norm(original_ankles[j] - dark[j]) for j in (0, 1))
                swapped = sum(np.linalg.norm(original_ankles[j] - dark[1 - j]) for j in (0, 1))
                if (min(direct, swapped) < 190
                        and np.linalg.norm(dark[0] - dark[1]) > 80
                        and np.linalg.norm(original_ankles[0] - original_ankles[1]) > 45):
                    duplicate[frame_index] = False
                    continue
            old = np.nanmean(result[frame_index, (15, 16), :2], axis=0)
            visual = []
            distractors = [result[frame_index, joint, :2]
                           for joint in (0, 5, 6)
                           if np.isfinite(result[frame_index, joint, :2]).all()]
            for candidate in candidates:
                point = candidate["ankle"]
                if any(np.linalg.norm(point - landmark) < 85 for landmark in distractors):
                    continue
                visual.append((point, 2.0 + min(candidate["area"], 1500) / 1500))
            for endpoint in dark:
                if any(np.linalg.norm(endpoint - point) < 95 for point, _ in visual):
                    continue
                visual.append((endpoint, 1.0))
            best_score = -np.inf
            pair = None
            for first, second in combinations(visual, 2):
                a, a_score = first
                b, b_score = second
                av, bv = a - hip[frame_index], b - hip[frame_index]
                a_dist, b_dist = np.linalg.norm(av), np.linalg.norm(bv)
                separation = np.linalg.norm(a - b)
                if (min(a_dist, b_dist) < .8 * leg_pixel or
                        max(a_dist, b_dist) > 650 or
                        separation < max(130, .8 * leg_pixel)):
                    continue
                cosine = np.dot(av, bv) / max(a_dist * b_dist, 1e-6)
                if cosine > .91:  # Two tips on the same visible leg.
                    continue
                near_old = min(np.linalg.norm(a - old), np.linalg.norm(b - old))
                score = (a_score + b_score + .4 * separation / leg_pixel
                         + .006 * (a_dist + b_dist)
                         - .012 * near_old)
                if score > best_score:
                    best_score, pair = score, [a, b]
            if pair is None:
                continue
            base, far = sorted(pair, key=lambda point: np.linalg.norm(point - old))
            old_ankles = original_ankles
            mismatch = min(
                np.linalg.norm(base - old_ankles[0]) + np.linalg.norm(far - old_ankles[1]),
                np.linalg.norm(base - old_ankles[1]) + np.linalg.norm(far - old_ankles[0]),
            )
            if duplicate[frame_index] or mismatch > max(160, .9 * leg_pixel):
                proposals[frame_index] = (base, far)
    finally:
        capture.release()

    corrected = 0
    changed_spans = 0
    isolated_repairs = 0
    start = 0
    while start < frames:
        if not duplicate[start]:
            start += 1
            continue
        end = start + 1
        while end < frames and duplicate[end]:
            end += 1
        available = [i for i in range(start, end) if i in proposals]
        if not available:
            start = end
            continue
        # Keep one anatomical side for each continuous collapse. Choose it by
        # the nearest non-collapsed ankle before and after the segment.
        costs = []
        for side in (15, 16):
            cost = 0.0
            for neighbor in (start - 1, end):
                if 0 <= neighbor < frames and np.isfinite(result[neighbor, side, :2]).all():
                    nearest_frame = available[0] if neighbor < start else available[-1]
                    cost += np.linalg.norm(proposals[nearest_frame][1] - result[neighbor, side, :2])
            costs.append(cost)
        far_side = 15 if costs[0] <= costs[1] else 16
        base_side = 16 if far_side == 15 else 15
        changed_spans += 1
        for frame_index in range(start, end):
            if frame_index in proposals:
                base, far = proposals[frame_index]
                result[frame_index, base_side, :2] = base
                result[frame_index, base_side, 2] = max(result[frame_index, base_side, 2], 0.7)
                result[frame_index, far_side, :2] = far
                result[frame_index, far_side, 2] = 0.8
                corrected += 1
            else:
                result[frame_index, far_side, :2] = np.nan
                result[frame_index, far_side, 2] = 0.0
            hip_joint = 11 if far_side == 15 else 12
            knee_joint = 13 if far_side == 15 else 14
            if frame_index in proposals and np.isfinite(result[frame_index, hip_joint, :2]).all():
                result[frame_index, knee_joint, :2] = (
                    result[frame_index, hip_joint, :2] + result[frame_index, far_side, :2]
                ) / 2
                result[frame_index, knee_joint, 2] = 0.65
            elif frame_index not in proposals:
                result[frame_index, knee_joint, :2] = np.nan
                result[frame_index, knee_joint, 2] = 0.0
        start = end
    for frame_index, pair in proposals.items():
        if duplicate[frame_index]:
            continue
        original = keypoints[frame_index, (15, 16), :2]
        candidate = np.stack(pair)
        direct = np.linalg.norm(candidate - original, axis=1)
        swapped = np.linalg.norm(candidate[::-1] - original, axis=1)
        assigned = candidate if direct.sum() <= swapped.sum() else candidate[::-1]
        distances = np.linalg.norm(assigned - original, axis=1)
        if min(distances) > 110 or max(distances) < max(150, .8 * leg_pixel):
            continue
        side = 15 + int(np.argmax(distances))
        hip_joint = 11 if side == 15 else 12
        knee_joint = 13 if side == 15 else 14
        result[frame_index, side, :2] = assigned[side - 15]
        result[frame_index, side, 2] = 0.8
        result[frame_index, knee_joint, :2] = (
            result[frame_index, hip_joint, :2] + result[frame_index, side, :2]
        ) / 2
        result[frame_index, knee_joint, 2] = 0.65
        isolated_repairs += 1
    return result, {
        "white_shoe_recovery": True,
        "collapsed_ankle_frames": int(np.count_nonzero(duplicate)),
        "shoe_corrected_frames": corrected,
        "shoe_corrected_spans": changed_spans,
        "shoe_isolated_repairs": isolated_repairs,
    }

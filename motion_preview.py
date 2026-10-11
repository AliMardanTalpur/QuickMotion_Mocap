"""Render a sampled camera and side view of Stage 2 joints for visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


EDGES_H36M17 = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
                (12, 13), (8, 14), (14, 15), (15, 16))
EDGES_SMPLX24 = ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
                 (0, 1), (1, 4), (4, 7), (7, 10),
                 (0, 2), (2, 5), (5, 8), (8, 11),
                 (9, 13), (13, 16), (16, 18), (18, 20),
                 (9, 14), (14, 17), (17, 19), (19, 21))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    points = np.load(args.input, allow_pickle=False)
    if points.ndim != 3 or points.shape[1:] not in ((17, 3), (24, 3)):
        parser.error("Input must contain (frames, 17/24, 3) points")
    edges = EDGES_SMPLX24 if points.shape[1] == 24 else EDGES_H36M17
    endpoints = (10, 11, 20, 21) if points.shape[1] == 24 else (3, 6, 13, 16)
    indices = np.linspace(0, len(points) - 1, args.samples).round().astype(int)
    rows = (args.samples + 3) // 4
    canvas = np.full((2 * rows * 260, 4 * 260, 3), 245, dtype=np.uint8)
    for number, frame_index in enumerate(indices):
        row, col = divmod(number, 4)
        for view, axes in enumerate(((0, 1), (2, 1))):
            tile = canvas[(row * 2 + view) * 260:(row * 2 + view + 1) * 260,
                          col * 260:(col + 1) * 260]
            pose = points[frame_index][:, axes].copy()
            pose -= pose[0]
            # A fixed metric scale makes anomalous body size immediately visible.
            pixels = (pose * 105 + np.array([130, 145])).round().astype(int)
            for parent, child in edges:
                cv2.line(tile, tuple(pixels[parent]), tuple(pixels[child]),
                         (20, 120, 40), 2, cv2.LINE_AA)
            for index, point in enumerate(pixels):
                cv2.circle(tile, tuple(point), 3,
                           (50, 40, 210) if index in endpoints else (30, 30, 30),
                           -1, cv2.LINE_AA)
            cv2.putText(tile, f"frame {frame_index + 1} {'camera' if view == 0 else 'side'}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .54, (0, 0, 0), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

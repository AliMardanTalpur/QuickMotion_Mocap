"""Versioned, pickle-free interchange formats between pipeline processes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def write_metadata(path: Path, **values: object) -> None:
    path.with_suffix(".json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, **values}, indent=2),
        encoding="utf-8",
    )


def read_metadata(path: Path) -> dict:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version in {path.with_suffix('.json')}")
    return metadata


@dataclass
class TrackSequence:
    frame_index: np.ndarray       # (N,), original video frame indices
    track_id: np.ndarray          # (N,), -1 means missing
    bbox_xyxy: np.ndarray         # (N,4), original image pixels
    crop_xyxy: np.ndarray         # (N,4), square ROI, may extend outside image
    crop_to_image: np.ndarray     # (N,3,3), crop pixels -> original image pixels
    keypoints_xyc: np.ndarray     # (N,17,3), original image pixels/confidence
    valid: np.ndarray             # (N,), a real tracker observation exists
    crop_valid: np.ndarray        # (N,), a crop exists (may bridge a short gap)

    def save(self, path: Path, **metadata: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.frame_index)
        if not all(len(getattr(self, name)) == n for name in (
            "track_id", "bbox_xyxy", "crop_xyxy", "crop_to_image",
            "keypoints_xyc", "valid", "crop_valid",
        )):
            raise ValueError("Track arrays have different frame counts")
        np.savez_compressed(path, **self.__dict__)
        write_metadata(path, kind="tracking", frames=n, **metadata)

    @classmethod
    def load(cls, path: Path) -> "TrackSequence":
        read_metadata(path)
        with np.load(path, allow_pickle=False) as data:
            result = cls(**{name: data[name] for name in cls.__dataclass_fields__})
        if result.bbox_xyxy.shape != (len(result.frame_index), 4):
            raise ValueError("Invalid tracking bbox shape")
        if result.keypoints_xyc.shape != (len(result.frame_index), 17, 3):
            raise ValueError("Invalid COCO-17 keypoint shape")
        return result


@dataclass
class SmplSequence:
    rotations: np.ndarray         # (N,24,3,3), SMPL local rotation matrices
    betas: np.ndarray             # (10,), one stable shape for the clip
    translation: np.ndarray       # (N,3), camera/world according to metadata
    joints: np.ndarray            # (N,24,3), same coordinate system as translation
    valid: np.ndarray             # (N,), true only for a model prediction
    confidence: np.ndarray        # (N,), track confidence proxy

    def save(self, path: Path, **metadata: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.valid)
        if (self.rotations.shape != (n, 24, 3, 3)
                or self.translation.shape != (n, 3)
                or self.joints.shape != (n, 24, 3)
                or self.betas.shape != (10,)):
            raise ValueError("Invalid SMPL array shapes")
        np.savez_compressed(path, **self.__dict__)
        write_metadata(path, kind="smpl", frames=n, **metadata)

    @classmethod
    def load(cls, path: Path) -> "SmplSequence":
        read_metadata(path)
        with np.load(path, allow_pickle=False) as data:
            result = cls(**{name: data[name] for name in cls.__dataclass_fields__})
        if result.rotations.shape != (len(result.valid), 24, 3, 3):
            raise ValueError("Invalid SMPL rotation shape")
        return result

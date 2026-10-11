"""Small contract and geometry checks without model downloads or GPU."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.contracts import SmplSequence, TrackSequence, read_metadata
from pipeline.crop import crop_square, square_roi
from pipeline.smooth import MotionFilter
from pipeline.contacts import FootContactCorrector

TEMP_PARENT = Path(__file__).resolve().parents[1] / "runs"
TEMP_PARENT.mkdir(exist_ok=True)


class PipelineTests(unittest.TestCase):
    def test_crop_transform_round_trip(self) -> None:
        frame = np.zeros((200, 300, 3), np.uint8)
        kp = np.full((17, 3), np.nan, np.float32)
        kp[:, 2] = 0
        kp[0] = (150, 20, .9)
        roi = square_roi(np.array([100, 50, 200, 180], np.float32), kp)
        image, transform = crop_square(frame, roi, 256)
        self.assertEqual(image.shape, (256, 256, 3))
        mapped = transform @ np.array([128, 128, 1], np.float32)
        np.testing.assert_allclose(mapped[:2], (roi[:2] + roi[2:]) / 2, atol=1e-4)

    def test_contract_and_filter(self) -> None:
        n = 20
        rotations = np.tile(np.eye(3, dtype=np.float32), (n, 24, 1, 1))
        angles = np.linspace(0, .7, n)
        rotations[:, 1] = Rotation.from_euler("z", angles[:, None]).as_matrix()
        rotations[8:10] = np.nan
        translation = np.zeros((n, 3), np.float32)
        translation[:, 0] = np.linspace(0, .3, n)
        translation[8:10] = np.nan
        joints = np.zeros((n, 24, 3), np.float32)
        joints[:, 15, 1] = 1.7
        joints[:, (7, 8), 1] = 0
        joints[8:10] = np.nan
        valid = np.ones(n, bool)
        valid[8:10] = False
        raw = SmplSequence(rotations, np.zeros(10, np.float32), translation,
                           joints, valid, valid.astype(np.float32))
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as temp:
            path = Path(temp) / "raw_smpl.npz"
            raw.save(path, model="synthetic", body_model="SMPL-24", fps=25)
            self.assertEqual(read_metadata(path)["fps"], 25)
            output = MotionFilter().run(SmplSequence.load(path))
            self.assertTrue(np.isfinite(output.rotations).all())
            self.assertTrue(np.isfinite(output.translation).all())
            self.assertFalse(output.valid[8])
            products = output.rotations @ output.rotations.transpose(0, 1, 3, 2)
            self.assertLess(float(np.max(np.abs(products - np.eye(3)))), 1e-5)
            corrected, report = FootContactCorrector().run(output, 25)
            self.assertEqual(len(corrected.valid), n)
            self.assertIn("contact_segments", report)

    def test_track_npz_without_pickle(self) -> None:
        n = 3
        sequence = TrackSequence(
            np.arange(n, dtype=np.int32), np.ones(n, np.int32),
            np.zeros((n, 4), np.float32), np.zeros((n, 4), np.float32),
            np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
            np.zeros((n, 17, 3), np.float32), np.ones(n, bool), np.ones(n, bool),
        )
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as temp:
            path = Path(temp) / "tracking.npz"
            sequence.save(path, fps=25, width=300, height=200)
            loaded = TrackSequence.load(path)
            self.assertEqual(loaded.keypoints_xyc.shape, (3, 17, 3))


if __name__ == "__main__":
    unittest.main()

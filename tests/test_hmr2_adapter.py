"""Exercise the HMR2 adapter contract with the official output field shapes."""

from __future__ import annotations

from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest

import cv2
import numpy as np
import torch

from pipeline.contracts import SmplSequence, TrackSequence
from pipeline.solver.hmr2_adapter import HMR2Solver


class FakeDataset:
    def __init__(self, cfg, image, boxes):
        self.size = image.shape[0]

    def __getitem__(self, index):
        return {"img": np.zeros((3, 256, 256), np.float32),
                "box_size": float(self.size)}


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        eye = torch.eye(3).reshape(1, 1, 3, 3)
        return {
            "pred_smpl_params": {
                "global_orient": eye,
                "body_pose": eye.repeat(1, 23, 1, 1),
                "betas": torch.zeros((1, 10)),
            },
            "pred_cam": torch.tensor([[1., 0., 0.]]),
            "pred_keypoints_3d": torch.zeros((1, 24, 3)),
        }


class AdapterTests(unittest.TestCase):
    def test_one_crop_contract(self):
        root = Path(__file__).resolve().parents[1] / "runs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            directory = Path(temp)
            repo = directory / "repo" / "hmr2"
            repo.mkdir(parents=True)
            checkpoint = directory / "model.ckpt"
            checkpoint.write_bytes(b"test fixture")
            crops = directory / "crops"
            crops.mkdir()
            cv2.imwrite(str(crops / "000000.png"), np.zeros((256, 256, 3), np.uint8))
            track = TrackSequence(
                np.array([0]), np.array([1]),
                np.array([[50, 50, 150, 150]], np.float32),
                np.array([[40, 40, 160, 160]], np.float32),
                np.eye(3, dtype=np.float32)[None],
                np.zeros((1, 17, 3), np.float32),
                np.array([True]), np.array([True]),
            )
            track_file = directory / "tracking.npz"
            track.save(track_file, fps=25, width=200, height=200)
            cfg = types.SimpleNamespace(
                EXTRA=types.SimpleNamespace(FOCAL_LENGTH=5000),
                MODEL=types.SimpleNamespace(IMAGE_SIZE=256),
            )
            modules = {
                "hmr2": types.ModuleType("hmr2"),
                "hmr2.models": types.ModuleType("hmr2.models"),
                "hmr2.datasets": types.ModuleType("hmr2.datasets"),
                "hmr2.datasets.vitdet_dataset": types.ModuleType("hmr2.datasets.vitdet_dataset"),
            }
            modules["hmr2.models"].load_hmr2 = lambda path: (FakeModel(), cfg)
            modules["hmr2.datasets.vitdet_dataset"].ViTDetDataset = FakeDataset
            original = {name: sys.modules.get(name) for name in modules}
            try:
                sys.modules.update(modules)
                output = directory / "raw_smpl.npz"
                HMR2Solver(repo.parent, checkpoint, "cpu").solve(track_file, output)
                motion = SmplSequence.load(output)
                self.assertEqual(motion.rotations.shape, (1, 24, 3, 3))
                self.assertTrue(np.isfinite(motion.translation).all())
                command = Path(__file__).resolve().parents[1] / "commands" / "03_filter_export.py"
                subprocess.run([sys.executable, str(command), "--run", str(directory)],
                               check=True, capture_output=True, text=True)
                self.assertTrue((directory / "motion_unity.npz").is_file())
            finally:
                for name, prior in original.items():
                    if prior is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = prior


if __name__ == "__main__":
    unittest.main()

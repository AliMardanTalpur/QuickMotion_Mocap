"""Run one HMR2 mesh model on tracked crops; save raw SMPL parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.solver.hmr2_adapter import HMR2Solver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--solver", choices=("hmr2",), default="hmr2")
    parser.add_argument("--hmr2-repo", type=Path, required=True,
                        help="Official 4DHumans checkout")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vram-limit-mb", type=int, default=3072)
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("The 4 GB hardware profile requires --batch-size 1")
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    HMR2Solver(args.hmr2_repo, args.checkpoint, device,
               args.vram_limit_mb).solve(args.run / "tracking.npz", args.run / "raw_smpl.npz")


if __name__ == "__main__":
    main()

"""Smooth SMPL rotations, reduce contact drift, and export Unity handoff NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.contacts import FootContactCorrector, GroundAligner
from pipeline.contracts import SmplSequence, read_metadata
from pipeline.smooth import MotionFilter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--foot-lock", choices=("on", "off"), default="on")
    args = parser.parse_args()
    raw_path = args.run / "raw_smpl.npz"
    meta = read_metadata(raw_path)
    raw = SmplSequence.load(raw_path)
    filtered = MotionFilter(args.sigma).run(raw)
    filtered, ground_report = GroundAligner().run(filtered)
    if args.foot_lock == "on":
        filtered, contact_report = FootContactCorrector().run(filtered, float(meta["fps"]))
    else:
        contact_report = {"contact_segments": 0, "foot_lock": "off"}
    output = args.output or args.run / "motion_unity.npz"
    filtered.save(
        output, model=meta["model"], body_model=meta["body_model"],
        fps=meta["fps"], source_raw=str(raw_path.resolve()),
        coordinate_system=meta["coordinate_system"],
        ground_aligned="estimated", world_trajectory_is_estimate=True,
        rotations="24 local SMPL rotation matrices; joint 0 is global orientation",
        translation="camera-derived root estimate, meters",
        joint_order="SMPL-24",
        invalid_frames="Filled from neighbors for continuity; consult valid array",
        ground=ground_report, contact=contact_report,
    )
    report = {
        "frames": len(filtered.valid), "fps": float(meta["fps"]),
        "observed_frames": int(filtered.valid.sum()),
        "solver": meta["model"], "vram_peak_reserved_mb": meta["vram_peak_reserved_mb"],
        **ground_report, **contact_report,
    }
    (args.run / "quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output} and {args.run / 'quality.json'}")


if __name__ == "__main__":
    main()

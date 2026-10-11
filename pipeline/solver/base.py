"""The solver process reads Stage 1 files and writes a SmplSequence."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class MeshSolver(Protocol):
    def solve(self, tracking_file: Path, output_file: Path) -> Path:
        ...

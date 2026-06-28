#!/usr/bin/env python3
"""Run the Step 1 learner projection script and write the local generated CSV."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
STEP1_CODE = BASE / "code/step1/step1_exact_boundary_projection_v2_merge_t2s_spanlen.py"
INPUT = BASE / "inputs/three_outputs_learners_charpairs_textcols.csv"
OUTPUT = BASE / "generated_csv/step1_projection.csv"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("step1_projection_local", STEP1_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Step 1 module from {STEP1_CODE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(STEP1_CODE),
            "--input",
            str(INPUT),
            "--output",
            str(OUTPUT),
        ]
        module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

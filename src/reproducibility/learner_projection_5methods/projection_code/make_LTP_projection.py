#!/usr/bin/env python3
from pathlib import Path

from baseline_from_combined import materialize_projection


HERE = Path(__file__).resolve().parent
BASE = HERE.parent

materialize_projection(
    BASE / "inputs/three_outputs_learners_charpairs_textcols.csv",
    BASE / "generated_csv/LTP_projection.csv",
    "LTP",
)

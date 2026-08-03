#!/usr/bin/env python3
from pathlib import Path

from baseline_from_combined import materialize_projection


HERE = Path(__file__).resolve().parent
BASE = HERE.parent

materialize_projection(
    BASE / "inputs/learners_three_outputs_add_ibm_only.csv",
    BASE / "generated_csv/IBM2_projection.csv",
    "IBM2",
)

#!/usr/bin/env python3
"""Materialize one baseline learner projection CSV from the combined table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHODS = {"IBM2", "LTP", "Levenshtein"}
FIELDS = ["Index", "Target_ID", "Gold_Source", "Target_Text", "Projected_Source"]


def materialize_projection(input_csv: Path, output_csv: Path, method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")

    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {input_csv}")
    index_col = "Index" if "Index" in rows[0] else "Source_ID"
    missing = [col for col in [index_col, "Target_ID", "Gold_Source", "Target_Text", method] if col not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {missing}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Index": row[index_col],
                    "Target_ID": row["Target_ID"],
                    "Gold_Source": row["Gold_Source"],
                    "Target_Text": row["Target_Text"],
                    "Projected_Source": row[method],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--input", type=Path, default=Path("../inputs/three_outputs_learners_charpairs_textcols.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize_projection(args.input, args.output, args.method)


if __name__ == "__main__":
    main()

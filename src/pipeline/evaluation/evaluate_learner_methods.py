#!/usr/bin/env python3
"""Evaluate stored learner-benchmark projection CSVs.

This script evaluates the paper-facing learner systems from the stored CSV
artifacts copied into src/reproducibility/learner_projection_5methods.
It does not rerun LTP, BERT, or glyph-table projection.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile

from jp_algorithm import evaluate, load_txt_file


REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE = REPO_ROOT / "src/reproducibility/learner_projection_5methods"
OUT = REPO_ROOT / "results/learner_benchmark/learner_eval_summary.csv"

METHODS = {
    "D": {
        "path": BUNDLE / "csv/LTP_projection.csv",
        "gold": "Gold_Source",
        "pred": "Projected_Source",
        "paper_name": "Direct segmentation",
    },
    "P1": {
        "path": BUNDLE / "csv/step1_projection.csv",
        "gold": "Source",
        "pred": "Projected_Source",
        "paper_name": "Identical-character projection",
    },
    "P2": {
        "path": BUNDLE / "csv/step2_projection.csv",
        "gold": "Source",
        "pred": "Projected_Source",
        "paper_name": "Similarity-based projection",
    },
}


def write_lines(lines: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).strip() + "\n")


def score_method(method: str, info: dict[str, str | Path]) -> dict[str, str | float | int]:
    csv_path = Path(info["path"])
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    gold_col = str(info["gold"])
    pred_col = str(info["pred"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        gold_path = tmp_dir / f"{method}.gold.txt"
        pred_path = tmp_dir / f"{method}.pred.txt"
        write_lines([row[gold_col] for row in rows], gold_path)
        write_lines([row[pred_col] for row in rows], pred_path)
        scores = evaluate(load_txt_file(str(gold_path)), load_txt_file(str(pred_path)))

    token = scores["Tokens"]
    sent = scores["Sentences"]
    return {
        "Method": method,
        "Paper_Name": str(info["paper_name"]),
        "Rows": len(rows),
        "Token_Precision": token.precision,
        "Token_Recall": token.recall,
        "Token_F1": token.f1,
        "Sentence_Precision": sent.precision,
        "Sentence_Recall": sent.recall,
        "Sentence_F1": sent.f1,
        "Source_File": str(csv_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [score_method(method, info) for method, info in METHODS.items()]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['Method']}: Token_F1={row['Token_F1']:.12f} rows={row['Rows']}")
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

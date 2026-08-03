#!/usr/bin/env python3
"""Evaluate generated learner projection CSVs in this folder."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "code/evaluation"))

from jp_algorithm import evaluate, load_txt_file  # noqa: E402


METHODS = {
    "IBM2": ("Gold_Source", "Projected_Source"),
    "LTP": ("Gold_Source", "Projected_Source"),
    "Levenshtein": ("Gold_Source", "Projected_Source"),
    "step1": ("Source", "Projected_Source"),
    "step2": ("Source", "Projected_Source"),
}
DATASET_NAMES = {
    "IBM2": "learners_three_outputs_add_ibm_only",
    "LTP": "three_outputs_learners_charpairs_textcols",
    "Levenshtein": "three_outputs_learners_charpairs_textcols",
    "step1": "learner_step1_exact_boundary_projection_v2_merge_t2s_spanlen",
    "step2": "learner_step2_output_final",
}
EVA_FIELDS = [
    "Dataset",
    "Method",
    "Token_Precision",
    "Token_Recall",
    "Token_F1",
    "Sentence_Precision",
    "Sentence_Recall",
    "Sentence_F1",
]


def write_txt(lines, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).strip() + "\n")


def evaluate_csv(method: str, csv_path: Path) -> dict[str, str | float]:
    gold_col, pred_col = METHODS[method]
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        gold_path = tmp_dir / f"{method}.gold.txt"
        pred_path = tmp_dir / f"{method}.pred.txt"
        write_txt([row[gold_col] for row in rows], gold_path)
        write_txt([row[pred_col] for row in rows], pred_path)
        scores = evaluate(load_txt_file(str(gold_path)), load_txt_file(str(pred_path)))

    token = scores["Tokens"]
    sentence = scores["Sentences"]
    return {
        "Dataset": DATASET_NAMES[method],
        "Method": method if method in {"IBM2", "LTP", "Levenshtein"} else "Projected_Source",
        "Token_Precision": token.precision,
        "Token_Recall": token.recall,
        "Token_F1": token.f1,
        "Sentence_Precision": sentence.precision,
        "Sentence_Recall": sentence.recall,
        "Sentence_F1": sentence.f1,
    }


def target_lineterminator(method: str) -> str:
    target = BASE / "eva" / f"eva_{method}.csv"
    if not target.exists():
        return "\n"
    data = target.read_bytes()
    return "\r\n" if b"\r\n" in data else "\n"


def main() -> None:
    out_dir = BASE / "generated_eva"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for method in METHODS:
        csv_path = BASE / "generated_csv" / f"{method}_projection.csv"
        if not csv_path.exists():
            print(f"Skipping {method}: {csv_path} does not exist")
            continue
        row = evaluate_csv(method, csv_path)
        summary_rows.append(row)
        eva_path = out_dir / f"eva_{method}.csv"
        with eva_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVA_FIELDS, lineterminator=target_lineterminator(method))
            writer.writeheader()
            writer.writerow(row)
        print(f"Wrote {eva_path}")

    with (out_dir / "combined_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVA_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {out_dir / 'combined_eval_summary.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge synthetic evaluation tables into one wide summary CSV.

Inputs:
  - data/0423/syn_step1_outputs/eva/eva_ALL.csv
  - data/0423/syn_step1_outputs/step1_synthetic_summary.csv
  - data/0423/syn_step2_outputs/eva/eva_ALL.csv
  - data/0423/syn_step2_outputs/step2_synthetic_summary.csv

Output:
  - data/0423/synthetic_eval_merged_summary.csv
  - data/0423/synthetic_eval_step1_summary.csv
  - data/0423/synthetic_eval_step2_summary.csv
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_noise(value: str) -> int:
    match = re.search(r"(\d{1,2})pct", value, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse noise pct from: {value}")
    return int(match.group(1))


def safe_ratio_str(num: str, den: str) -> str:
    if not num or not den:
        return ""
    den_value = float(den)
    if den_value == 0:
        return "0"
    return str(float(num) / den_value)


def build_step1_summary_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        index[Path(row["output_file"]).stem] = row
    return index


def build_step2_summary_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset = Path(row["file"]).stem + "_step2_projection"
        index[dataset] = row
    return index


def merge_rows(
    eval_rows: list[dict[str, str]],
    summary_index: dict[str, dict[str, str]],
    stage: str,
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []

    for eval_row in eval_rows:
        dataset = eval_row["Dataset"]
        summary_row = summary_index.get(dataset, {})

        merged.append(
            {
                "Stage": stage,
                "Noise_Pct": str(extract_noise(dataset)),
                "Dataset": dataset,
                "Method": eval_row["Method"],
                "Input_File": summary_row.get("input_file", summary_row.get("file", "")),
                "Output_File": summary_row.get("output_file", dataset + ".csv"),
                "Rows": summary_row.get("rows", ""),
                "Total_Chars": summary_row.get("total_chars", ""),
                "Step1_Aligned": summary_row.get("step1_aligned", summary_row.get("total_aligned", "")),
                "Step1_Aligned_Ratio": summary_row.get(
                    "step1_aligned_ratio",
                    summary_row.get(
                        "total_aligned_ratio",
                        safe_ratio_str(
                            summary_row.get("total_aligned", ""),
                            summary_row.get("total_chars", ""),
                        ),
                    ),
                ),
                "Step2_Aligned": summary_row.get("step2_aligned", "0" if stage == "step1" else ""),
                "Step2_Aligned_Ratio": summary_row.get(
                    "step2_aligned_ratio", "0" if stage == "step1" else ""
                ),
                "Total_Aligned": summary_row.get("total_aligned", ""),
                "Total_Aligned_Ratio": summary_row.get("total_aligned_ratio", ""),
                "Total_Unaligned": summary_row.get(
                    "total_unaligned", summary_row.get("still_unaligned_src", "")
                ),
                "Still_Unaligned_Src": summary_row.get(
                    "still_unaligned_src", summary_row.get("total_unaligned", "")
                ),
                "Still_Unaligned_Src_Ratio": summary_row.get(
                    "still_unaligned_src_ratio",
                    summary_row.get(
                        "total_unaligned_ratio",
                        safe_ratio_str(
                            summary_row.get("total_unaligned", ""),
                            summary_row.get("total_chars", ""),
                        ),
                    ),
                ),
                "Still_Unaligned_Tgt": summary_row.get("still_unaligned_tgt", ""),
                "Projected_Boundaries": summary_row.get(
                    "total_projected_boundaries",
                    summary_row.get("projected_boundaries", ""),
                ),
                "Skipped_Boundaries": summary_row.get(
                    "total_skipped_boundaries",
                    summary_row.get("skipped_boundaries", ""),
                ),
                "Merged_Spans": summary_row.get(
                    "total_merged_spans", summary_row.get("merged_spans", "")
                ),
                "Token_Precision": eval_row["Token_Precision"],
                "Token_Recall": eval_row["Token_Recall"],
                "Token_F1": eval_row["Token_F1"],
                "Sentence_Precision": eval_row["Sentence_Precision"],
                "Sentence_Recall": eval_row["Sentence_Recall"],
                "Sentence_F1": eval_row["Sentence_F1"],
            }
        )

    merged.sort(key=lambda row: (row["Stage"], int(row["Noise_Pct"]), row["Method"]))
    return merged


def to_step1_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered = [row for row in rows if row["Stage"] == "step1"]
    out = []
    for row in filtered:
        out.append(
            {
                "Noise_Pct": row["Noise_Pct"],
                "Dataset": row["Dataset"],
                "Method": row["Method"],
                "Input_File": row["Input_File"],
                "Output_File": row["Output_File"],
                "Total_Chars": row["Total_Chars"],
                "Total_Aligned": row["Total_Aligned"],
                "Total_Unaligned": row["Total_Unaligned"],
                "Projected_Boundaries": row["Projected_Boundaries"],
                "Skipped_Boundaries": row["Skipped_Boundaries"],
                "Merged_Spans": row["Merged_Spans"],
                "Token_Precision": row["Token_Precision"],
                "Token_Recall": row["Token_Recall"],
                "Token_F1": row["Token_F1"],
                "Sentence_Precision": row["Sentence_Precision"],
                "Sentence_Recall": row["Sentence_Recall"],
                "Sentence_F1": row["Sentence_F1"],
            }
        )
    return out


def to_step2_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered = [row for row in rows if row["Stage"] == "step2"]
    out = []
    for row in filtered:
        out.append(
            {
                "Noise_Pct": row["Noise_Pct"],
                "Dataset": row["Dataset"],
                "Method": row["Method"],
                "Input_File": row["Input_File"],
                "Output_File": row["Output_File"],
                "Rows": row["Rows"],
                "Total_Chars": row["Total_Chars"],
                "Step1_Aligned": row["Step1_Aligned"],
                "Step1_Aligned_Ratio": row["Step1_Aligned_Ratio"],
                "Step2_Aligned": row["Step2_Aligned"],
                "Step2_Aligned_Ratio": row["Step2_Aligned_Ratio"],
                "Total_Aligned": row["Total_Aligned"],
                "Total_Aligned_Ratio": row["Total_Aligned_Ratio"],
                "Still_Unaligned_Src": row["Still_Unaligned_Src"],
                "Still_Unaligned_Src_Ratio": row["Still_Unaligned_Src_Ratio"],
                "Still_Unaligned_Tgt": row["Still_Unaligned_Tgt"],
                "Projected_Boundaries": row["Projected_Boundaries"],
                "Skipped_Boundaries": row["Skipped_Boundaries"],
                "Merged_Spans": row["Merged_Spans"],
                "Token_Precision": row["Token_Precision"],
                "Token_Recall": row["Token_Recall"],
                "Token_F1": row["Token_F1"],
                "Sentence_Precision": row["Sentence_Precision"],
                "Sentence_Recall": row["Sentence_Recall"],
                "Sentence_F1": row["Sentence_F1"],
            }
        )
    return out


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    base_dir = (script_dir / "../data/0423").resolve()

    step1_eval = read_csv(base_dir / "syn_step1_outputs/eva/eva_ALL.csv")
    step2_eval = read_csv(base_dir / "syn_step2_outputs/eva/eva_ALL.csv")
    step1_summary = build_step1_summary_index(
        read_csv(base_dir / "syn_step1_outputs/step1_synthetic_summary.csv")
    )
    step2_summary = build_step2_summary_index(
        read_csv(base_dir / "syn_step2_outputs/step2_synthetic_summary.csv")
    )

    merged_rows = []
    merged_rows.extend(merge_rows(step1_eval, step1_summary, "step1"))
    merged_rows.extend(merge_rows(step2_eval, step2_summary, "step2"))
    merged_rows.sort(key=lambda row: (row["Stage"], int(row["Noise_Pct"]), row["Method"]))

    fieldnames = [
        "Stage",
        "Noise_Pct",
        "Dataset",
        "Method",
        "Input_File",
        "Output_File",
        "Rows",
        "Total_Chars",
        "Step1_Aligned",
        "Step1_Aligned_Ratio",
        "Step2_Aligned",
        "Step2_Aligned_Ratio",
        "Total_Aligned",
        "Total_Aligned_Ratio",
        "Total_Unaligned",
        "Still_Unaligned_Src",
        "Still_Unaligned_Src_Ratio",
        "Still_Unaligned_Tgt",
        "Projected_Boundaries",
        "Skipped_Boundaries",
        "Merged_Spans",
        "Token_Precision",
        "Token_Recall",
        "Token_F1",
        "Sentence_Precision",
        "Sentence_Recall",
        "Sentence_F1",
    ]

    out_path = base_dir / "synthetic_eval_merged_summary.csv"
    write_csv(out_path, merged_rows, fieldnames)
    print(f"Saved merged summary: {out_path}")

    step1_out_path = base_dir / "synthetic_eval_step1_summary.csv"
    step1_rows = to_step1_rows(merged_rows)
    write_csv(
        step1_out_path,
        step1_rows,
        [
            "Noise_Pct",
            "Dataset",
            "Method",
            "Input_File",
            "Output_File",
            "Total_Chars",
            "Total_Aligned",
            "Total_Unaligned",
            "Projected_Boundaries",
            "Skipped_Boundaries",
            "Merged_Spans",
            "Token_Precision",
            "Token_Recall",
            "Token_F1",
            "Sentence_Precision",
            "Sentence_Recall",
            "Sentence_F1",
        ],
    )
    print(f"Saved step1 summary: {step1_out_path}")

    step2_out_path = base_dir / "synthetic_eval_step2_summary.csv"
    step2_rows = to_step2_rows(merged_rows)
    write_csv(
        step2_out_path,
        step2_rows,
        [
            "Noise_Pct",
            "Dataset",
            "Method",
            "Input_File",
            "Output_File",
            "Rows",
            "Total_Chars",
            "Step1_Aligned",
            "Step1_Aligned_Ratio",
            "Step2_Aligned",
            "Step2_Aligned_Ratio",
            "Total_Aligned",
            "Total_Aligned_Ratio",
            "Still_Unaligned_Src",
            "Still_Unaligned_Src_Ratio",
            "Still_Unaligned_Tgt",
            "Projected_Boundaries",
            "Skipped_Boundaries",
            "Merged_Spans",
            "Token_Precision",
            "Token_Recall",
            "Token_F1",
            "Sentence_Precision",
            "Sentence_Recall",
            "Sentence_F1",
        ],
    )
    print(f"Saved step2 summary: {step2_out_path}")


if __name__ == "__main__":
    main()

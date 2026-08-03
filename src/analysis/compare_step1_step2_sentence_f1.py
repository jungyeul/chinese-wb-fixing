#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from pathlib import Path

from jp_algorithm import evaluate, load_txt


NOISE_RE = re.compile(r"(\d{1,2})pct", re.IGNORECASE)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_noise(name: str) -> int:
    match = NOISE_RE.search(name)
    if not match:
        raise ValueError(f"Cannot parse noise pct from: {name}")
    return int(match.group(1))


def build_ud_from_sentence(sent: str):
    return load_txt(io.StringIO((sent or "").strip() + "\n"))


def score_sentence(gold: str, pred: str) -> tuple[float, float, float, float, float, float]:
    scores = evaluate(build_ud_from_sentence(gold), build_ud_from_sentence(pred))
    token = scores["Tokens"]
    sent = scores["Sentences"]
    return (
        token.precision,
        token.recall,
        token.f1,
        sent.precision,
        sent.recall,
        sent.f1,
    )


def classify_delta(delta: float, eps: float = 1e-12) -> str:
    if delta > eps:
        return "improved"
    if delta < -eps:
        return "worsened"
    return "same"


def dataset_key(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("_step1_projection", "")
    stem = stem.replace("_step2_projection", "")
    return stem


def summarize_group(rows: list[dict[str, str]]) -> dict[str, str]:
    counts = Counter(row["Change"] for row in rows)
    return {
        "Rows": str(len(rows)),
        "Improved_Rows": str(counts["improved"]),
        "Worsened_Rows": str(counts["worsened"]),
        "Same_Rows": str(counts["same"]),
        "Mean_Step1_Token_F1": str(sum(float(r["Step1_Token_F1"]) for r in rows) / len(rows)),
        "Mean_Step2_Token_F1": str(sum(float(r["Step2_Token_F1"]) for r in rows) / len(rows)),
        "Mean_Token_F1_Delta": str(sum(float(r["Token_F1_Delta"]) for r in rows) / len(rows)),
        "Total_Positive_Delta": str(sum(max(float(r["Token_F1_Delta"]), 0.0) for r in rows)),
        "Total_Negative_Delta": str(sum(min(float(r["Token_F1_Delta"]), 0.0) for r in rows)),
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    base_dir = (script_dir / "../data/0423").resolve()
    step1_dir = base_dir / "syn_step1_outputs"
    step2_dir = base_dir / "syn_step2_outputs"

    step1_files = {
        dataset_key(path): path
        for path in step1_dir.glob("*_step1_projection.csv")
    }
    step2_files = {
        dataset_key(path): path
        for path in step2_dir.glob("*_step2_projection.csv")
    }

    common_keys = sorted(step1_files.keys() & step2_files.keys(), key=extract_noise)
    if not common_keys:
        raise FileNotFoundError("No matching step1/step2 projection files found.")

    detail_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for key in common_keys:
        step1_rows = read_csv(step1_files[key])
        step2_rows = read_csv(step2_files[key])
        if len(step1_rows) != len(step2_rows):
            raise ValueError(f"Row count mismatch for {key}: {len(step1_rows)} vs {len(step2_rows)}")

        group_rows: list[dict[str, str]] = []
        for idx, (row1, row2) in enumerate(zip(step1_rows, step2_rows), start=1):
            gold = row1["Source"]
            step1_pred = row1["Projected_Source"]
            step2_pred = row2["Projected_Source"]

            s1_tp, s1_tr, s1_tf, s1_sp, s1_sr, s1_sf = score_sentence(gold, step1_pred)
            s2_tp, s2_tr, s2_tf, s2_sp, s2_sr, s2_sf = score_sentence(gold, step2_pred)

            delta = s2_tf - s1_tf
            out_row = {
                "Noise_Pct": str(extract_noise(key)),
                "Dataset": key,
                "Row_Index": str(idx),
                "ID": row1.get("ID", row1.get("CTB_ID", "")),
                "Step1_Token_Precision": str(s1_tp),
                "Step1_Token_Recall": str(s1_tr),
                "Step1_Token_F1": str(s1_tf),
                "Step2_Token_Precision": str(s2_tp),
                "Step2_Token_Recall": str(s2_tr),
                "Step2_Token_F1": str(s2_tf),
                "Token_F1_Delta": str(delta),
                "Change": classify_delta(delta),
                "Step1_Sentence_F1": str(s1_sf),
                "Step2_Sentence_F1": str(s2_sf),
                "Step1_Projected_Source": step1_pred,
                "Step2_Projected_Source": step2_pred,
                "Gold_Source": gold,
            }
            detail_rows.append(out_row)
            group_rows.append(out_row)

        group_summary = {
            "Noise_Pct": str(extract_noise(key)),
            "Dataset": key,
            **summarize_group(group_rows),
        }
        summary_rows.append(group_summary)

    detail_rows.sort(key=lambda row: (int(row["Noise_Pct"]), int(row["Row_Index"])))
    summary_rows.sort(key=lambda row: int(row["Noise_Pct"]))

    detail_fields = [
        "Noise_Pct",
        "Dataset",
        "Row_Index",
        "ID",
        "Step1_Token_Precision",
        "Step1_Token_Recall",
        "Step1_Token_F1",
        "Step2_Token_Precision",
        "Step2_Token_Recall",
        "Step2_Token_F1",
        "Token_F1_Delta",
        "Change",
        "Step1_Sentence_F1",
        "Step2_Sentence_F1",
        "Gold_Source",
        "Step1_Projected_Source",
        "Step2_Projected_Source",
    ]
    summary_fields = [
        "Noise_Pct",
        "Dataset",
        "Rows",
        "Improved_Rows",
        "Worsened_Rows",
        "Same_Rows",
        "Mean_Step1_Token_F1",
        "Mean_Step2_Token_F1",
        "Mean_Token_F1_Delta",
        "Total_Positive_Delta",
        "Total_Negative_Delta",
    ]

    detail_path = base_dir / "step1_vs_step2_sentence_token_f1_detail.csv"
    summary_path = base_dir / "step1_vs_step2_sentence_token_f1_summary.csv"
    write_csv(detail_path, detail_rows, detail_fields)
    write_csv(summary_path, summary_rows, summary_fields)
    print(f"Saved detail report: {detail_path}")
    print(f"Saved summary report: {summary_path}")


if __name__ == "__main__":
    main()

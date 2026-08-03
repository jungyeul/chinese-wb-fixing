#!/usr/bin/env python3
"""Write the paper-facing synthetic F1 table from stored result data."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT = REPO_ROOT / "results/synthetic/synthetic_all_methods_report.csv"
OUTPUT = REPO_ROOT / "results/synthetic/synthetic_token_f1_table.csv"


def method_label(row: pd.Series) -> str:
    dataset = str(row["Dataset"])
    method = str(row["Method"])
    if method == "LTP":
        return "D"
    if "step1_projection" in dataset:
        return "P1"
    if "step2_projection" in dataset:
        return "P2"
    return method


def noise_level(dataset: str) -> int:
    match = re.search(r"charnoise_(\d+)pct", dataset)
    if not match:
        raise ValueError(f"Cannot parse noise level from {dataset}")
    return int(match.group(1))


def main() -> None:
    df = pd.read_csv(INPUT)
    df = df.copy()
    df["Noise"] = df["Dataset"].map(noise_level)
    df["Paper_Method"] = df.apply(method_label, axis=1)
    df = df[df["Paper_Method"].isin(["D", "P1", "P2"])]
    table = (
        df.pivot_table(index="Noise", columns="Paper_Method", values="Token_F1", aggfunc="first")
        .reset_index()
        .sort_values("Noise")
    )
    table = table[["Noise", "D", "P1", "P2"]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(table.to_string(index=False))
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute source- and target-side character alignment coverage for P1/P2.

The projection CSVs store alignments as target-index -> source-index pairs,
for example:

    2->2:在 3->3:冰
    [E]2->2:在 [S]14->14:对

Coverage is computed separately on the source and target sides:

    source coverage = aligned unique source character indices / source chars
    target coverage = aligned unique target character indices / target chars

Spaces in tokenized sentences are removed before counting characters.
Punctuation is counted as normal characters.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ALIGNMENT_RE = re.compile(r"(?:\[(?:E|S)\])?(\d+)->(\d+):")


def parse_alignments(
    alignment_string: object,
    source_len: int,
    target_len: int,
) -> tuple[set[int], set[int]]:
    """Return valid aligned source and target character index sets."""

    source_indices: set[int] = set()
    target_indices: set[int] = set()

    if pd.isna(alignment_string):
        return source_indices, target_indices

    for match in ALIGNMENT_RE.finditer(str(alignment_string)):
        target_idx = int(match.group(1))
        source_idx = int(match.group(2))

        if 0 <= source_idx < source_len:
            source_indices.add(source_idx)
        if 0 <= target_idx < target_len:
            target_indices.add(target_idx)

    return source_indices, target_indices


def untokenize(text: object) -> str:
    """Remove tokenization spaces before character counting."""

    return str(text).replace(" ", "")


def compute_coverage(input_path: Path, method: str) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute corpus-level and sentence-level coverage for one method."""

    df = pd.read_csv(input_path)

    total_source_chars = 0
    total_target_chars = 0
    aligned_source_chars = 0
    aligned_target_chars = 0
    sentence_rows: list[dict[str, object]] = []

    for row_index, row in df.iterrows():
        source = untokenize(row["Source"])
        target = untokenize(row["Target_Text"])

        source_indices, target_indices = parse_alignments(
            row.get("Alignments", ""),
            source_len=len(source),
            target_len=len(target),
        )

        total_source_chars += len(source)
        total_target_chars += len(target)
        aligned_source_chars += len(source_indices)
        aligned_target_chars += len(target_indices)

        sentence_rows.append(
            {
                "method": method,
                "row_index": row_index,
                "sentence_id": row.get("ID", row.get("Index", row_index + 1)),
                "source_chars": len(source),
                "target_chars": len(target),
                "aligned_source_chars": len(source_indices),
                "aligned_target_chars": len(target_indices),
                "source_coverage": len(source_indices) / len(source) if source else 0.0,
                "target_coverage": len(target_indices) / len(target) if target else 0.0,
            }
        )

    summary = {
        "method": method,
        "input_file": str(input_path),
        "sentences": len(df),
        "aligned_source_chars": aligned_source_chars,
        "total_source_chars": total_source_chars,
        "source_coverage": aligned_source_chars / total_source_chars,
        "source_coverage_percent": 100 * aligned_source_chars / total_source_chars,
        "aligned_target_chars": aligned_target_chars,
        "total_target_chars": total_target_chars,
        "target_coverage": aligned_target_chars / total_target_chars,
        "target_coverage_percent": 100 * aligned_target_chars / total_target_chars,
    }

    return summary, pd.DataFrame(sentence_rows)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    projection_files = {
        "P1": base_dir / "csv" / "step1_projection.csv",
        "P2": base_dir / "csv" / "step2_projection.csv",
    }

    summaries: list[dict[str, object]] = []

    for method, input_path in projection_files.items():
        summary, sentence_df = compute_coverage(input_path, method)
        summaries.append(summary)
        sentence_df.to_csv(base_dir / f"alignment_coverage_{method}_by_sentence.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(base_dir / "alignment_coverage_p1_p2_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

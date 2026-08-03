#!/usr/bin/env python3
"""Count source characters involved in P2 word-order variation links.

This script treats a P2 residual link as a word-order variation link when:

1. the link is marked as [S], i.e. it was added by Step 2 similarity-based
   residual alignment rather than Step 1 exact matching; and
2. the link is non-monotone with respect to at least one other aligned
   character pair in the same sentence.

The alignment strings have the form:

    [E]2->2:在 [S]14->14:对

where the index order is target-index -> source-index.

The main reported number is the number of unique source-side character
positions participating in such non-monotone [S] links, divided by the total
number of source-side characters in the learner benchmark.

For reference, the script also reports a broader count that includes all
source-side characters in crossing links, including exact [E] anchors crossed
by a residual [S] link. The paper-facing number should normally use the main
residual-[S] count.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ALIGNMENT_RE = re.compile(r"\[(?P<kind>E|S)\](?P<tgt>\d+)->(?P<src>\d+):")


def untokenize(text: object) -> str:
    """Remove tokenization spaces before character counting."""

    return str(text).replace(" ", "")


def parse_alignments(alignment_string: object, source_len: int, target_len: int) -> list[tuple[str, int, int]]:
    """Parse valid P2 alignments as (kind, target_index, source_index)."""

    alignments: list[tuple[str, int, int]] = []
    for match in ALIGNMENT_RE.finditer(str(alignment_string)):
        kind = match.group("kind")
        target_idx = int(match.group("tgt"))
        source_idx = int(match.group("src"))
        if 0 <= source_idx < source_len and 0 <= target_idx < target_len:
            alignments.append((kind, target_idx, source_idx))
    return alignments


def crossing_source_sets(
    alignments: list[tuple[str, int, int]],
) -> tuple[set[int], set[int], set[int], set[int]]:
    """Return source/target sets for non-monotone residual links and all crossings."""

    residual_crossing_source: set[int] = set()
    residual_crossing_target: set[int] = set()
    all_crossing_source: set[int] = set()
    all_crossing_target: set[int] = set()

    for i, (kind_i, target_i, source_i) in enumerate(alignments):
        for kind_j, target_j, source_j in alignments[i + 1 :]:
            # Non-monotone/crossing when target order and source order disagree.
            if (target_i - target_j) * (source_i - source_j) >= 0:
                continue

            all_crossing_source.update({source_i, source_j})
            all_crossing_target.update({target_i, target_j})

            if kind_i == "S":
                residual_crossing_source.add(source_i)
                residual_crossing_target.add(target_i)
            if kind_j == "S":
                residual_crossing_source.add(source_j)
                residual_crossing_target.add(target_j)

    return (
        residual_crossing_source,
        residual_crossing_target,
        all_crossing_source,
        all_crossing_target,
    )


def compute_word_order_coverage(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute corpus-level and sentence-level P2 word-order coverage."""

    df = pd.read_csv(input_path)

    total_source_chars = 0
    total_target_chars = 0
    p2_aligned_source_chars = 0
    p2_aligned_target_chars = 0
    p2_similarity_source_chars = 0
    p2_similarity_target_chars = 0
    word_order_similarity_source_chars = 0
    word_order_similarity_target_chars = 0
    word_order_all_crossing_source_chars = 0
    word_order_all_crossing_target_chars = 0
    sentence_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        source = untokenize(row["Source"])
        target = untokenize(row["Target_Text"])
        alignments = parse_alignments(row.get("Alignments", ""), len(source), len(target))

        aligned_source = {source_idx for _, _, source_idx in alignments}
        aligned_target = {target_idx for _, target_idx, _ in alignments}
        similarity_source = {source_idx for kind, _, source_idx in alignments if kind == "S"}
        similarity_target = {target_idx for kind, target_idx, _ in alignments if kind == "S"}

        (
            residual_crossing_source,
            residual_crossing_target,
            all_crossing_source,
            all_crossing_target,
        ) = crossing_source_sets(alignments)

        total_source_chars += len(source)
        total_target_chars += len(target)
        p2_aligned_source_chars += len(aligned_source)
        p2_aligned_target_chars += len(aligned_target)
        p2_similarity_source_chars += len(similarity_source)
        p2_similarity_target_chars += len(similarity_target)
        word_order_similarity_source_chars += len(residual_crossing_source)
        word_order_similarity_target_chars += len(residual_crossing_target)
        word_order_all_crossing_source_chars += len(all_crossing_source)
        word_order_all_crossing_target_chars += len(all_crossing_target)

        if residual_crossing_source:
            sentence_rows.append(
                {
                    "Index": row.get("Index", ""),
                    "ID": row.get("ID", ""),
                    "Source": row["Source"],
                    "Target_Text": row["Target_Text"],
                    "source_chars_in_word_order_S_links": "".join(
                        source[i] for i in sorted(residual_crossing_source)
                    ),
                    "source_positions_in_word_order_S_links": " ".join(
                        str(i) for i in sorted(residual_crossing_source)
                    ),
                    "target_positions_in_word_order_S_links": " ".join(
                        str(i) for i in sorted(residual_crossing_target)
                    ),
                    "num_source_chars_in_word_order_S_links": len(residual_crossing_source),
                    "all_crossing_source_positions_including_exact_anchors": " ".join(
                        str(i) for i in sorted(all_crossing_source)
                    ),
                }
            )

    summary = {
        "setting": "P2",
        "definition_main": (
            "source characters in P2 residual [S] links that cross at least "
            "one other aligned link"
        ),
        "total_source_chars": total_source_chars,
        "total_target_chars": total_target_chars,
        "p2_aligned_source_chars": p2_aligned_source_chars,
        "p2_aligned_target_chars": p2_aligned_target_chars,
        "p2_similarity_source_chars": p2_similarity_source_chars,
        "p2_similarity_target_chars": p2_similarity_target_chars,
        "word_order_similarity_source_chars": word_order_similarity_source_chars,
        "word_order_similarity_target_chars": word_order_similarity_target_chars,
        "word_order_all_crossing_source_chars": word_order_all_crossing_source_chars,
        "word_order_all_crossing_target_chars": word_order_all_crossing_target_chars,
        "sentences_with_word_order_similarity_links": len(sentence_rows),
        "word_order_similarity_source_percent_of_all_source": (
            100 * word_order_similarity_source_chars / total_source_chars
        ),
        "word_order_similarity_source_percent_of_p2_aligned_source": (
            100 * word_order_similarity_source_chars / p2_aligned_source_chars
        ),
        "word_order_all_crossing_source_percent_of_all_source": (
            100 * word_order_all_crossing_source_chars / total_source_chars
        ),
        "word_order_all_crossing_source_percent_of_p2_aligned_source": (
            100 * word_order_all_crossing_source_chars / p2_aligned_source_chars
        ),
    }

    return pd.DataFrame([summary]), pd.DataFrame(sentence_rows)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "csv" / "step2_projection.csv"

    summary_df, sentence_df = compute_word_order_coverage(input_path)

    summary_path = base_dir / "p2_word_order_alignment_coverage_summary.csv"
    sentence_path = base_dir / "p2_word_order_alignment_links_by_sentence.csv"

    summary_df.to_csv(summary_path, index=False)
    sentence_df.to_csv(sentence_path, index=False)

    print(summary_df.T.to_string(header=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {sentence_path}")


if __name__ == "__main__":
    main()

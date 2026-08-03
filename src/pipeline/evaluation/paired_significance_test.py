#!/usr/bin/env python3
"""Paired significance tests for learner benchmark token F1.

The script reconstructs sentence-level token counts from the official
projection CSVs, checks that corpus F1 matches the paper numbers, then runs:

- paired sentence bootstrap confidence intervals for F1 differences
- paired approximate randomization tests over sentence-level outcomes
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parents[1]
FILES = {
    "D": BASE / "csv/LTP_projection.csv",
    "P1": BASE / "csv/step1_projection.csv",
    "P2": BASE / "csv/step2_projection.csv",
}
GOLD_COL = {"D": "Gold_Source", "P1": "Source", "P2": "Source"}
PRED_COL = {"D": "Projected_Source", "P1": "Projected_Source", "P2": "Projected_Source"}
PAIRS = [("D", "P1"), ("P1", "P2"), ("D", "P2")]


def token_spans(segmented: str) -> set[tuple[int, int]]:
    spans = set()
    pos = 0
    for token in segmented.strip().split():
        spans.add((pos, pos + len(token)))
        pos += len(token)
    return spans


def sentence_counts(path: Path, gold_col: str, pred_col: str) -> np.ndarray:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    counts = []
    for row_idx, row in enumerate(rows, start=1):
        gold = str(row[gold_col]).strip()
        pred = str(row[pred_col]).strip()
        if gold.replace(" ", "") != pred.replace(" ", ""):
            raise ValueError(f"Raw string mismatch in {path} row {row_idx}")
        gold_spans = token_spans(gold)
        pred_spans = token_spans(pred)
        counts.append((len(gold_spans & pred_spans), len(pred_spans), len(gold_spans)))
    return np.array(counts, dtype=np.int64)


def corpus_f1(counts: np.ndarray) -> float:
    correct, pred_total, gold_total = counts.sum(axis=0)
    return 2 * correct / (pred_total + gold_total)


def main() -> None:
    rng = np.random.default_rng(20260522)
    iterations = 50_000

    counts = {
        method: sentence_counts(FILES[method], GOLD_COL[method], PRED_COL[method])
        for method in FILES
    }

    print("Method,Correct,Predicted,Gold,Token_F1")
    for method, method_counts in counts.items():
        correct, pred_total, gold_total = method_counts.sum(axis=0)
        print(f"{method},{correct},{pred_total},{gold_total},{corpus_f1(method_counts):.12f}")

    n_sentences = len(next(iter(counts.values())))
    print(f"\nSentences={n_sentences}, Iterations={iterations}")
    print("Comparison,F1_A,F1_B,Delta,Delta_pct_points,Bootstrap_CI_low,Bootstrap_CI_high,ApproxRand_p_two_sided")

    for method_a, method_b in PAIRS:
        a = counts[method_a]
        b = counts[method_b]
        observed_delta = corpus_f1(b) - corpus_f1(a)

        boot = np.empty(iterations)
        for i in range(iterations):
            sample = rng.integers(0, n_sentences, n_sentences)
            boot[i] = corpus_f1(b[sample]) - corpus_f1(a[sample])
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

        extreme = 0
        for _ in range(iterations):
            swap = rng.integers(0, 2, n_sentences).astype(bool)
            x = np.where(swap[:, None], a, b)
            y = np.where(swap[:, None], b, a)
            randomized_delta = corpus_f1(y) - corpus_f1(x)
            if abs(randomized_delta) >= abs(observed_delta) - 1e-15:
                extreme += 1
        p_value = (extreme + 1) / (iterations + 1)

        print(
            f"{method_a}->{method_b},"
            f"{corpus_f1(a):.12f},"
            f"{corpus_f1(b):.12f},"
            f"{observed_delta:.12f},"
            f"{observed_delta * 100:.6f},"
            f"{ci_low:.12f},"
            f"{ci_high:.12f},"
            f"{p_value:.8f}"
        )


if __name__ == "__main__":
    main()

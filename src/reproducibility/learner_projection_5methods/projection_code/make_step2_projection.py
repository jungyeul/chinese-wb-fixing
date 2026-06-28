#!/usr/bin/env python3
"""Run the Step 2 full-best learner projection and write the local generated CSV.

This mirrors the full-best setting used for the stored Step 2 CSV:
embedding=0.1, shape=0.6, pinyin=0.0, position=0.3, threshold=0.85.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
STEP2_CODE = BASE / "code/step2/step2_ltp_4features_t2s_projection_spanlen.py"
INPUT = BASE / "inputs/learner_corpus.csv"
SHAPE = BASE / "inputs/triplet_no_dup_threshold.csv"
OUTPUT = BASE / "generated_csv/step2_projection.csv"

FULL_WEIGHTS = {
    "embedding": 0.1,
    "shape": 0.6,
    "pinyin": 0.0,
    "position": 0.3,
}
THRESHOLD = 0.85


def load_step2_module():
    spec = importlib.util.spec_from_file_location("step2_projection_local", STEP2_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Step 2 module from {STEP2_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_step2_module()
    original_fn = module.align_similar_chars_4feat_t2s

    def align_with_fixed_weights(source, target, unaligned_src, unaligned_tgt, emb_sim_matrix, shape_table, threshold=0.5):
        new_alignments = {}
        used_src = set()
        for tgt_idx in unaligned_tgt:
            tgt_char = target[tgt_idx]
            best_src_idx = -1
            best_score = threshold
            for src_idx in unaligned_src:
                if src_idx in used_src:
                    continue
                src_char = source[src_idx]
                emb_sim = emb_sim_matrix[src_idx, tgt_idx]
                shape_sim = shape_table.get((src_char, tgt_char), 0.0)
                py_sim = module.pinyin_similarity(src_char, tgt_char)
                pos_diff = abs(src_idx - tgt_idx) / max(len(source), len(target))
                pos_sim = 1 - pos_diff
                score = (
                    FULL_WEIGHTS["embedding"] * emb_sim
                    + FULL_WEIGHTS["shape"] * shape_sim
                    + FULL_WEIGHTS["pinyin"] * py_sim
                    + FULL_WEIGHTS["position"] * pos_sim
                )
                if score > best_score:
                    best_score = score
                    best_src_idx = src_idx
            if best_src_idx != -1:
                new_alignments[tgt_idx] = best_src_idx
                used_src.add(best_src_idx)
        return new_alignments

    module.align_similar_chars_4feat_t2s = align_with_fixed_weights
    try:
        module.load_bert_model("bert-base-chinese")
        shape_table = module.load_shape_table(str(SHAPE))
        sep = module.detect_separator(str(INPUT))
        df = pd.read_csv(INPUT, sep=sep, encoding="utf-8-sig")

        results = []
        for idx, row in df.iterrows():
            source = row["Source_Text"]
            target = row["Target_Text"]
            result = module.process_pair(source, target, shape_table, THRESHOLD)
            results.append(
                {
                    "Index": row.get("Index", idx),
                    "ID": "",
                    "Source": source,
                    "Target_Text": target,
                    "Base_Segmentation": result["ltp_seg"],
                    "Alignments": result["align_str"],
                    "Projected_Boundary_Count": len(result["projected_boundaries"]),
                    "Skipped_Boundaries": " ".join(result["skipped_boundaries"]),
                    "Merged_Spans": " ".join(result["merged_spans"]),
                    "Merge_Skips": " ".join(result["merge_skips"]),
                    "Projected_Source": result["projected"],
                }
            )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(OUTPUT, index=False)
    finally:
        module.align_similar_chars_4feat_t2s = original_fn


if __name__ == "__main__":
    main()

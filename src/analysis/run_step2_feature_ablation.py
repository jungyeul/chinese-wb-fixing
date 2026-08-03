#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src/pipeline/evaluation"))

from jp_algorithm import evaluate, load_txt_file  # noqa: E402


STEP2_PATH = REPO_ROOT / "src/pipeline/alignment/step2_similarity_alignment_projection.py"
INPUT_CSV = REPO_ROOT / "data/learner_benchmark/levcos_human_selected_pairs.csv"
SHAPE_CSV = REPO_ROOT / "data/local_resources/triplet_no_dup_threshold.csv"
OUTPUT_DIR = REPO_ROOT / "results/ablation/step2_feature_ablation_outputs/generated"

FULL_WEIGHTS = {
    "embedding": 0.1,
    "shape": 0.6,
    "pinyin": 0.0,
    "position": 0.3,
}
FULL_THRESHOLD = 0.85

ABLATIONS = [
    ("full_best", FULL_WEIGHTS),
    ("wo_embedding", {**FULL_WEIGHTS, "embedding": 0.0}),
    ("wo_shape", {**FULL_WEIGHTS, "shape": 0.0}),
    ("wo_pinyin", {**FULL_WEIGHTS, "pinyin": 0.0}),
    ("wo_position", {**FULL_WEIGHTS, "position": 0.0}),
]


def load_step2_module():
    spec = importlib.util.spec_from_file_location("step2_ablation_base", STEP2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {STEP2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def detect_separator(filepath: Path) -> str:
    with open(filepath, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


def run_projection(module, weights: dict[str, float], output_csv: Path) -> None:
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
                    weights["embedding"] * emb_sim
                    + weights["shape"] * shape_sim
                    + weights["pinyin"] * py_sim
                    + weights["position"] * pos_sim
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
        shape_table = module.load_shape_table(str(SHAPE_CSV))

        sep = detect_separator(INPUT_CSV)
        df = pd.read_csv(INPUT_CSV, sep=sep, encoding="utf-8-sig")

        results = []
        for idx, row in df.iterrows():
            source = row["Source_Text"]
            target = row["Target_Text"]
            result = module.process_pair(source, target, shape_table, FULL_THRESHOLD)
            results.append(
                {
                    "Index": row.get("Index", idx),
                    "ID": row.get("Target_ID", ""),
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

        pd.DataFrame(results).to_csv(output_csv, index=False)
    finally:
        module.align_similar_chars_4feat_t2s = original_fn


def evaluate_output(csv_path: Path) -> dict[str, float | str]:
    dataset_name = csv_path.stem
    out_dir = csv_path.parent / f"eva_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    gold_txt = out_dir / f"{dataset_name}.gold.txt"
    pred_txt = out_dir / f"{dataset_name}.pred.txt"

    with gold_txt.open("w", encoding="utf-8") as f:
        for sent in df["Source"]:
            f.write(str(sent).strip() + "\n")
    with pred_txt.open("w", encoding="utf-8") as f:
        for sent in df["Projected_Source"]:
            f.write(str(sent).strip() + "\n")

    scores = evaluate(load_txt_file(str(gold_txt)), load_txt_file(str(pred_txt)))
    token = scores["Tokens"]
    sent = scores["Sentences"]
    return {
        "Dataset": dataset_name,
        "Token_Precision": token.precision,
        "Token_Recall": token.recall,
        "Token_F1": token.f1,
        "Sentence_Precision": sent.precision,
        "Sentence_Recall": sent.recall,
        "Sentence_F1": sent.f1,
    }


def main() -> None:
    if not SHAPE_CSV.exists():
        raise FileNotFoundError(
            f"Glyph similarity table is required for this rerun: {SHAPE_CSV}. "
            "See docs/DATA_PREPARATION.md for the expected schema."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    module = load_step2_module()
    summary_rows = []

    for name, weights in ABLATIONS:
        output_csv = OUTPUT_DIR / f"learner_step2_{name}.csv"
        print(f"Running {name} with weights={weights} threshold={FULL_THRESHOLD}")
        run_projection(module, weights, output_csv)
        metrics = evaluate_output(output_csv)
        metrics.update(
            {
                "w_embedding": weights["embedding"],
                "w_shape": weights["shape"],
                "w_pinyin": weights["pinyin"],
                "w_position": weights["position"],
                "threshold": FULL_THRESHOLD,
            }
        )
        summary_rows.append(metrics)
        print(
            f"  -> Token F1={metrics['Token_F1']:.10f} "
            f"(P={metrics['Token_Precision']:.10f}, R={metrics['Token_Recall']:.10f})"
        )

    summary_path = OUTPUT_DIR / "step2_feature_ablation_eval_summary.csv"
    fieldnames = [
        "Dataset",
        "w_embedding",
        "w_shape",
        "w_pinyin",
        "w_position",
        "threshold",
        "Token_Precision",
        "Token_Recall",
        "Token_F1",
        "Sentence_Precision",
        "Sentence_Recall",
        "Sentence_F1",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

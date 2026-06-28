#!/usr/bin/env python3
"""
Optimize Step 2 feature weights and threshold for token-level F1.

This script keeps the Step 2 projection logic unchanged, but replaces the fixed
equal weights:

    embedding=0.25, shape=0.25, pinyin=0.25, position=0.25

with a grid search over:

    w_embedding + w_shape + w_pinyin + w_position = 1
    threshold

The objective is the same token-level F1 used by learners_eva.py, computed
in-memory from token spans for speed.
"""

import argparse
import importlib.util
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
STEP2_PATH = SCRIPT_DIR / "step2_ltp_4features_t2s_projection_spanlen.py"


def import_step2_module():
    spec = importlib.util.spec_from_file_location("step2_spanlen", STEP2_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_spans(segmented_text):
    spans = set()
    pos = 0
    for token in str(segmented_text).strip().split():
        start = pos
        end = pos + len(token)
        spans.add((start, end))
        pos = end
    return spans


def token_f1(gold_segmentations, pred_segmentations):
    tp = 0
    pred_total = 0
    gold_total = 0
    sent_exact = 0

    for gold, pred in zip(gold_segmentations, pred_segmentations):
        gold_spans = token_spans(gold)
        pred_spans = token_spans(pred)
        tp += len(gold_spans & pred_spans)
        pred_total += len(pred_spans)
        gold_total += len(gold_spans)
        sent_exact += int(gold_spans == pred_spans)

    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sent_f1 = sent_exact / len(gold_segmentations) if len(gold_segmentations) else 0.0
    return precision, recall, f1, sent_f1


def generate_weight_grid(step):
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-9:
        raise ValueError("--weight-step must divide 1.0 exactly, e.g. 0.25, 0.2, 0.1, 0.05")

    weights = []
    for emb in range(units + 1):
        for shape in range(units - emb + 1):
            for pinyin in range(units - emb - shape + 1):
                position = units - emb - shape - pinyin
                weights.append(
                    {
                        "embedding": emb / units,
                        "shape": shape / units,
                        "pinyin": pinyin / units,
                        "position": position / units,
                    }
                )
    return weights


def generate_thresholds(start, stop, step):
    values = []
    cur = start
    while cur <= stop + 1e-9:
        values.append(round(cur, 10))
        cur += step
    return values


def source_column(df):
    if "Source" in df.columns:
        return "Source"
    if "Gold_Source" in df.columns:
        return "Gold_Source"
    if "Source_Text" in df.columns:
        return "Source_Text"
    raise ValueError("Cannot find source column: expected Source, Gold_Source, or Source_Text")


def build_feature_cache(df, shape_table, step2, model_name, cache_path=None):
    if cache_path and cache_path.exists():
        print(f"Loading feature cache: {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    step2.load_bert_model(model_name)
    src_col = source_column(df)
    has_ltp_source = "LTP_Source" in df.columns

    cache = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building feature cache"):
        source = str(row[src_col])
        target = str(row["Target_Text"])
        source_chars = source.replace(" ", "")
        target_chars = target.replace(" ", "")

        if not source_chars:
            cache.append(
                {
                    "source": source,
                    "target": target,
                    "source_chars": source_chars,
                    "target_chars": target_chars,
                    "base_boundaries": step2.get_word_final_positions(source),
                    "step1_aligned": {},
                    "candidate_features": {},
                }
            )
            continue

        ltp_seg = str(row["LTP_Source"]) if has_ltp_source else step2.ltp_segment(source)
        base_boundaries = step2.get_word_final_positions(ltp_seg)
        step1_aligned, unaligned_src, unaligned_tgt = step2.exact_alignment_ins_del_only(
            source_chars, target_chars
        )
        emb_sim_matrix = step2.compute_embedding_similarity(source_chars, target_chars)

        candidate_features = {}
        for tgt_idx in unaligned_tgt:
            tgt_char = target_chars[tgt_idx]
            candidates = []
            for src_idx in unaligned_src:
                src_char = source_chars[src_idx]
                pos_diff = abs(src_idx - tgt_idx) / max(len(source_chars), len(target_chars))
                candidates.append(
                    (
                        src_idx,
                        float(emb_sim_matrix[src_idx, tgt_idx]),
                        float(shape_table.get((src_char, tgt_char), 0.0)),
                        float(step2.pinyin_similarity(src_char, tgt_char)),
                        float(1 - pos_diff),
                    )
                )
            candidate_features[tgt_idx] = candidates

        cache.append(
            {
                "source": source,
                "target": target,
                "source_chars": source_chars,
                "target_chars": target_chars,
                "base_boundaries": base_boundaries,
                "step1_aligned": step1_aligned,
                "candidate_features": candidate_features,
            }
        )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(cache, f)
        print(f"Saved feature cache: {cache_path}")

    return cache


def align_with_weights(candidate_features, weights, threshold):
    new_alignments = {}
    used_src = set()

    for tgt_idx in sorted(candidate_features):
        best_src_idx = -1
        best_score = threshold

        for src_idx, emb_sim, shape_sim, py_sim, pos_sim in candidate_features[tgt_idx]:
            if src_idx in used_src:
                continue

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


def project_from_cached(item, weights, threshold, step2):
    source_chars = item["source_chars"]
    if not source_chars:
        return item["source"]

    step2_aligned = align_with_weights(item["candidate_features"], weights, threshold)
    all_aligned = {**item["step1_aligned"], **step2_aligned}

    final_word_final, _, _ = step2.project_boundaries_conservatively(
        target_text=item["target"],
        aligned=all_aligned,
        base_boundaries=item["base_boundaries"],
    )
    final_word_final, _, _ = step2.apply_exact_match_merge_constraint(
        target_text=item["target"],
        aligned=all_aligned,
        word_final_positions=final_word_final,
    )
    final_word_final.add(len(source_chars) - 1)
    return step2.positions_to_segmented(source_chars, final_word_final)


def main():
    parser = argparse.ArgumentParser(description="Optimize Step 2 feature weights and threshold")
    parser.add_argument("--input", required=True, help="Input CSV with Source/Gold_Source and Target_Text")
    parser.add_argument("--shape", required=True, help="Shape similarity table CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for optimization outputs")
    parser.add_argument("--model", default="bert-base-chinese", help="BERT model name")
    parser.add_argument("--weight-step", type=float, default=0.1, help="Grid step for weights")
    parser.add_argument("--threshold-start", type=float, default=0.3)
    parser.add_argument("--threshold-stop", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--cache", default=None, help="Optional pickle cache path for features")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top rows to save separately")
    args = parser.parse_args()

    step2 = import_step2_module()
    shape_table = step2.load_shape_table(args.shape)
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    src_col = source_column(df)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else output_dir / "step2_feature_cache.pkl"

    cache = build_feature_cache(df, shape_table, step2, args.model, cache_path)
    gold = df[src_col].astype(str).tolist()

    weight_grid = generate_weight_grid(args.weight_step)
    thresholds = generate_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)
    print(f"Weight combinations: {len(weight_grid)}")
    print(f"Thresholds: {len(thresholds)}")
    print(f"Total trials: {len(weight_grid) * len(thresholds)}")

    results = []
    best = None
    best_preds = None

    for weights in tqdm(weight_grid, desc="Searching weights"):
        for threshold in thresholds:
            preds = [project_from_cached(item, weights, threshold, step2) for item in cache]
            precision, recall, f1, sent_f1 = token_f1(gold, preds)
            row = {
                "w_embedding": weights["embedding"],
                "w_shape": weights["shape"],
                "w_pinyin": weights["pinyin"],
                "w_position": weights["position"],
                "threshold": threshold,
                "Token_Precision": precision,
                "Token_Recall": recall,
                "Token_F1": f1,
                "Sentence_F1": sent_f1,
            }
            results.append(row)

            if best is None or f1 > best["Token_F1"]:
                best = row
                best_preds = preds

    results_df = pd.DataFrame(results).sort_values("Token_F1", ascending=False).reset_index(drop=True)
    results_path = output_dir / "weight_search_results.csv"
    top_path = output_dir / "weight_search_top.csv"
    best_output_path = output_dir / "best_step2_output.csv"

    results_df.to_csv(results_path, index=False)
    results_df.head(args.top_k).to_csv(top_path, index=False)

    best_df = df.copy()
    best_df["Projected_Source"] = best_preds
    best_df.to_csv(best_output_path, index=False)

    print("\n=== Best setting ===")
    print(pd.DataFrame([best]).to_string(index=False))
    print(f"\nSaved all results: {results_path}")
    print(f"Saved top results: {top_path}")
    print(f"Saved best output: {best_output_path}")


if __name__ == "__main__":
    main()

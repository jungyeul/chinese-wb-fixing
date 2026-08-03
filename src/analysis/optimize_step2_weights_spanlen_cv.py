#!/usr/bin/env python3
"""
5-fold cross validation for Step 2 feature-weight optimization.

This uses the same cached feature representation as
optimize_step2_weights_spanlen.py. For each weight/threshold setting, it
computes token-level F1 on each fold and ranks settings by mean fold F1.
"""

import argparse
import importlib.util
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
OPT_PATH = SCRIPT_DIR / "optimize_step2_weights_spanlen.py"
STEP2_PATH = SCRIPT_DIR / "step2_ltp_4features_t2s_projection_spanlen.py"


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_folds(n_items, n_folds, seed):
    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    folds = [[] for _ in range(n_folds)]
    for i, idx in enumerate(indices):
        folds[i % n_folds].append(idx)
    return [sorted(fold) for fold in folds]


def score_subset(opt, gold, preds, indices):
    subset_gold = [gold[i] for i in indices]
    subset_preds = [preds[i] for i in indices]
    return opt.token_f1(subset_gold, subset_preds)


def main():
    parser = argparse.ArgumentParser(description="5-fold CV for Step 2 weight search")
    parser.add_argument("--input", required=True, help="Input CSV with Source/Gold_Source and Target_Text")
    parser.add_argument("--shape", required=True, help="Shape similarity table CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for CV outputs")
    parser.add_argument("--model", default="bert-base-chinese", help="BERT model name")
    parser.add_argument("--weight-step", type=float, default=0.1, help="Grid step for weights")
    parser.add_argument("--threshold-start", type=float, default=0.3)
    parser.add_argument("--threshold-stop", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cache", default=None, help="Optional feature cache path")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    opt = import_module(OPT_PATH, "optimize_step2_weights_spanlen")
    step2 = import_module(STEP2_PATH, "step2_spanlen")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    src_col = opt.source_column(df)
    gold = df[src_col].astype(str).tolist()

    shape_table = step2.load_shape_table(args.shape)
    cache_path = Path(args.cache) if args.cache else output_dir / "step2_feature_cache.pkl"
    cache = opt.build_feature_cache(df, shape_table, step2, args.model, cache_path)

    folds = make_folds(len(df), args.folds, args.seed)
    fold_rows = []
    for fold_id, indices in enumerate(folds):
        for idx in indices:
            fold_rows.append({"row_index": idx, "ID": df.iloc[idx].get("ID", idx), "fold": fold_id})
    pd.DataFrame(fold_rows).sort_values("row_index").to_csv(output_dir / "cv_folds.csv", index=False)

    weight_grid = opt.generate_weight_grid(args.weight_step)
    thresholds = opt.generate_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)
    print(f"Weight combinations: {len(weight_grid)}")
    print(f"Thresholds: {len(thresholds)}")
    print(f"Total trials: {len(weight_grid) * len(thresholds)}")
    print(f"Folds: {args.folds}, seed={args.seed}")

    results = []
    per_fold_results = []
    best = None
    best_preds = None

    for weights in tqdm(weight_grid, desc="CV searching weights"):
        for threshold in thresholds:
            preds = [opt.project_from_cached(item, weights, threshold, step2) for item in cache]

            fold_f1s = []
            fold_precisions = []
            fold_recalls = []
            fold_sentence_f1s = []
            for fold_id, indices in enumerate(folds):
                precision, recall, f1, sent_f1 = score_subset(opt, gold, preds, indices)
                fold_f1s.append(f1)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                fold_sentence_f1s.append(sent_f1)
                per_fold_results.append(
                    {
                        "w_embedding": weights["embedding"],
                        "w_shape": weights["shape"],
                        "w_pinyin": weights["pinyin"],
                        "w_position": weights["position"],
                        "threshold": threshold,
                        "fold": fold_id,
                        "Token_Precision": precision,
                        "Token_Recall": recall,
                        "Token_F1": f1,
                        "Sentence_F1": sent_f1,
                    }
                )

            full_precision, full_recall, full_f1, full_sent_f1 = opt.token_f1(gold, preds)
            row = {
                "w_embedding": weights["embedding"],
                "w_shape": weights["shape"],
                "w_pinyin": weights["pinyin"],
                "w_position": weights["position"],
                "threshold": threshold,
                "CV_Token_F1_Mean": sum(fold_f1s) / len(fold_f1s),
                "CV_Token_F1_Min": min(fold_f1s),
                "CV_Token_F1_Max": max(fold_f1s),
                "CV_Token_Precision_Mean": sum(fold_precisions) / len(fold_precisions),
                "CV_Token_Recall_Mean": sum(fold_recalls) / len(fold_recalls),
                "CV_Sentence_F1_Mean": sum(fold_sentence_f1s) / len(fold_sentence_f1s),
                "Full_Token_Precision": full_precision,
                "Full_Token_Recall": full_recall,
                "Full_Token_F1": full_f1,
                "Full_Sentence_F1": full_sent_f1,
            }
            results.append(row)

            if best is None or row["CV_Token_F1_Mean"] > best["CV_Token_F1_Mean"]:
                best = row
                best_preds = preds

    results_df = pd.DataFrame(results).sort_values(
        ["CV_Token_F1_Mean", "CV_Token_F1_Min"], ascending=False
    ).reset_index(drop=True)
    per_fold_df = pd.DataFrame(per_fold_results)

    results_path = output_dir / "cv_weight_search_results.csv"
    top_path = output_dir / "cv_weight_search_top.csv"
    per_fold_path = output_dir / "cv_weight_search_per_fold.csv"
    best_output_path = output_dir / "cv_best_step2_output.csv"

    results_df.to_csv(results_path, index=False)
    results_df.head(args.top_k).to_csv(top_path, index=False)
    per_fold_df.to_csv(per_fold_path, index=False)

    best_df = df.copy()
    best_df["Projected_Source"] = best_preds
    best_df.to_csv(best_output_path, index=False)

    print("\n=== Best CV setting ===")
    print(pd.DataFrame([best]).to_string(index=False))
    print(f"\nSaved CV results: {results_path}")
    print(f"Saved top CV results: {top_path}")
    print(f"Saved per-fold results: {per_fold_path}")
    print(f"Saved best CV output: {best_output_path}")


if __name__ == "__main__":
    main()

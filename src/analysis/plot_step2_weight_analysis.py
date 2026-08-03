#!/usr/bin/env python3
"""
Plot Step 2 weight-search diagnostics.

Outputs:
- feature pairwise correlation matrix from cached candidate features
- mean/max F1 by individual feature weight
- pairwise heatmaps of max F1 over two selected weights
"""

import argparse
import pickle
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FEATURES = ["embedding", "shape", "pinyin", "position"]
WEIGHT_COLS = [f"w_{name}" for name in FEATURES]


def load_candidate_features(cache_path, max_pairs=None):
    rows = []
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    for item in cache:
        for tgt_idx, candidates in item["candidate_features"].items():
            for src_idx, emb, shape, pinyin, position in candidates:
                rows.append(
                    {
                        "embedding": emb,
                        "shape": shape,
                        "pinyin": pinyin,
                        "position": position,
                    }
                )
                if max_pairs and len(rows) >= max_pairs:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def save_corr_plot(feature_df, out_dir):
    corr = feature_df[FEATURES].corr(method="pearson")
    corr_path = out_dir / "feature_correlation.csv"
    corr.to_csv(corr_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=35, ha="right")
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature Correlation (Candidate Pairs)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def metric_col(results_df):
    if "CV_Token_F1_Mean" in results_df.columns:
        return "CV_Token_F1_Mean"
    return "Token_F1"


def save_single_weight_plots(results_df, out_dir, metric):
    summary_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    axes = axes.ravel()

    for ax, weight_col in zip(axes, WEIGHT_COLS):
        grouped = (
            results_df.groupby(weight_col)[metric]
            .agg(["mean", "max", "count"])
            .reset_index()
            .sort_values(weight_col)
        )
        grouped.insert(0, "weight", weight_col)
        summary_rows.append(grouped)

        ax.plot(grouped[weight_col], grouped["mean"], marker="o", label="mean")
        ax.plot(grouped[weight_col], grouped["max"], marker="s", label="max")
        ax.set_title(weight_col)
        ax.set_xlabel("weight value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel(metric)
    axes[2].set_ylabel(metric)
    fig.suptitle(f"{metric} by Individual Weight")
    fig.tight_layout()
    fig.savefig(out_dir / "single_weight_effects.png", dpi=200)
    plt.close(fig)

    pd.concat(summary_rows, ignore_index=True).to_csv(out_dir / "single_weight_effects.csv", index=False)


def save_pairwise_heatmaps(results_df, out_dir, metric):
    pair_dir = out_dir / "pairwise_heatmaps"
    pair_dir.mkdir(parents=True, exist_ok=True)

    for x_col, y_col in combinations(WEIGHT_COLS, 2):
        pivot = results_df.pivot_table(
            index=y_col,
            columns=x_col,
            values=metric,
            aggfunc="max",
        ).sort_index(ascending=True)
        pivot.to_csv(pair_dir / f"heatmap_{x_col}_vs_{y_col}.csv")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{y:.1f}" for y in pivot.index])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Max {metric}: {x_col} vs {y_col}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(pair_dir / f"heatmap_{x_col}_vs_{y_col}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Step 2 weight diagnostics")
    parser.add_argument("--search-results", required=True, help="weight_search_results.csv or cv_weight_search_results.csv")
    parser.add_argument("--feature-cache", required=True, help="step2_feature_cache.pkl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pairs", type=int, default=500000, help="Max candidate pairs for feature correlation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(args.search_results)
    metric = metric_col(results_df)

    feature_df = load_candidate_features(Path(args.feature_cache), args.max_pairs)
    feature_df.to_csv(out_dir / "feature_candidate_sample.csv", index=False)

    save_corr_plot(feature_df, out_dir)
    save_single_weight_plots(results_df, out_dir, metric)
    save_pairwise_heatmaps(results_df, out_dir, metric)

    print(f"Metric plotted: {metric}")
    print(f"Feature pairs used for correlation: {len(feature_df)}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()

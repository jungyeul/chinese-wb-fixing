#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


SERIES_ORDER = [
    ("step2", "Projected_Source", "Step2", "#9467bd"),
    ("step1", "Projected_Source", "Step1", "#d62728"),
    ("baseline", "IBM2", "IBM2", "#1f77b4"),
    ("baseline", "LTP", "LTP", "#4c78a8"),
    ("baseline", "Levenshtein", "Levenshtein", "#72b7b2"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    report_path = (script_dir / "../data/0423/synthetic_all_methods_report.csv").resolve()
    out_path = report_path.with_name("token_f1_all_methods_trend.png")

    rows = read_csv(report_path)
    by_key: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        key = (row["Stage"], row["Method"])
        by_key.setdefault(key, {})[int(row["Noise_Pct"])] = float(row["Token_F1"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    point_labels: dict[int, list[dict[str, float | str]]] = {}

    for stage, method, label, color in SERIES_ORDER:
        key = (stage, method)
        points = by_key.get(key, {})
        xs = sorted(points)
        ys = [points[x] for x in xs]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=color,
            label=label,
        )

        if xs and ys:
            for x, y in zip(xs, ys):
                point_labels.setdefault(x, []).append(
                    {
                        "label": label,
                        "color": color,
                        "x": x,
                        "y": y,
                    }
                )

    for x, labels in sorted(point_labels.items()):
        labels.sort(key=lambda item: item["y"])
        min_gap = 0.007
        adjusted = []
        for item in labels:
            y = item["y"]
            if adjusted and y - adjusted[-1]["y_adj"] < min_gap:
                y = adjusted[-1]["y_adj"] + min_gap
            adjusted.append({**item, "y_adj": y})

        upper = 0.995
        for i in range(len(adjusted) - 1, -1, -1):
            y = adjusted[i]["y_adj"]
            if y > upper:
                y = upper
            adjusted[i]["y_adj"] = y
            upper = y - min_gap

        x_offset = 0.12 if x < 10 else 0.16
        for item in adjusted:
            ax.annotate(
                f"{item['y'] * 100:.2f}%",
                (item["x"], item["y"]),
                xytext=(item["x"] + x_offset, item["y_adj"]),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=8.5,
                color=item["color"],
                arrowprops={
                    "arrowstyle": "-",
                    "color": item["color"],
                    "lw": 1.0,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "alpha": 0.8,
                },
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )

    ax.set_title("Token F1 Across Char Noise Levels", fontsize=18, pad=16)
    ax.set_xlabel("Noise Level", fontsize=13)
    ax.set_ylabel("Token F1", fontsize=13)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"{i}pct" for i in range(1, 11)])
    ax.set_xlim(0.8, 10.95)
    ax.set_ylim(0.82, 1.0)
    ax.legend(frameon=True, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()

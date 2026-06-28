#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


SERIES_ORDER = [
    ("baseline", "LTP", r"$D$", "#2F5DA8", "o"),
    ("step1", "Projected_Source", r"$P_1$", "#D62728", "s"),
    ("step2", "Projected_Source", r"$P_2$", "#7E57C2", "^"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    report_path = (script_dir / "../data/0423/synthetic_all_methods_report.csv").resolve()
    out_png = report_path.with_name("synthetic_token_f1_4methods_paper.png")
    out_pdf = report_path.with_name("synthetic_token_f1_4methods_paper.pdf")
    out_png_3methods = report_path.with_name("synthetic_token_f1_3methods_paper.png")
    out_pdf_3methods = report_path.with_name("synthetic_token_f1_3methods_paper.pdf")

    rows = read_csv(report_path)
    by_key: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        key = (row["Stage"], row["Method"])
        by_key.setdefault(key, {})[int(row["Noise_Pct"])] = float(row["Token_F1"])

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.8,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.4,
            "ytick.labelsize": 10.4,
            "legend.fontsize": 10.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.65, 2.42))

    for stage, method, label, color, marker in SERIES_ORDER:
        points = by_key[(stage, method)]
        xs = sorted(points)
        ys = [points[x] for x in xs]
        ax.plot(
            xs,
            ys,
            marker=marker,
            linewidth=2.2,
            markersize=5.0,
            markeredgewidth=0,
            color=color,
            label=label,
        )

    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Token $F_1$")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"{i}%" for i in range(1, 11)])
    ax.set_yticks([0.84, 0.88, 0.92, 0.96, 0.99])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", length=3.2, width=0.8, color="#3a3a3a")
    ax.set_xlim(0.8, 10.2)
    ax.set_ylim(0.825, 0.995)
    ax.legend(
        frameon=True,
        loc="lower left",
        borderpad=0.28,
        labelspacing=0.2,
        handlelength=1.55,
        handletextpad=0.45,
        framealpha=0.92,
        edgecolor="#d0d0d0",
        facecolor="white",
        ncol=1,
    )
    ax.grid(axis="y", color="#d6d6d6", alpha=0.75, linewidth=0.62)
    ax.grid(axis="x", color="#e7e7e7", alpha=0.55, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.16, right=0.995, top=0.985, bottom=0.18)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png_3methods, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf_3methods, bbox_inches="tight")
    print(f"Saved figure: {out_png}")
    print(f"Saved figure: {out_pdf}")
    print(f"Saved figure: {out_png_3methods}")
    print(f"Saved figure: {out_pdf_3methods}")


if __name__ == "__main__":
    main()

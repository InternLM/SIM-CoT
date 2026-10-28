#!/usr/bin/env python3
"""Draw the README results figure from the reviewed paper data in source.json.

Only Matplotlib and the Python standard library are required. Run this file
from any directory, or import and call results() from make_visuals.py.
"""

from decimal import Decimal
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


OUT = Path(__file__).resolve().parent
BASELINE = "#79b6df"
SIMCOT = "#83c9c2"
INK = "#242424"
MUTED = "#646464"
RULE = "#d9d8d3"
BACKGROUND = "#ffffff"


def results():
    """Save results-overview.png and results-overview.svg beside this script."""
    source = json.loads((OUT / "source.json").read_text(encoding="utf-8"))
    rows = source["results"]
    efficiency = source["token_efficiency"]

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "svg.hashsalt": "simcot-paper-results",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(14, 8.5), dpi=150)
        fig.patch.set_facecolor(BACKGROUND)
        ax.set_facecolor(BACKGROUND)
        fig.subplots_adjust(left=.31, right=.82, top=.76, bottom=.26)

        fig.text(
            .055, .925, "GSM8k-Aug results", fontsize=29,
            fontfamily="DejaVu Serif", color=INK,
        )
        fig.text(
            .055, .872,
            "Matched baseline and baseline + SIM-CoT comparisons · Paper Tables 1–3",
            fontsize=14, color=MUTED,
        )
        fig.legend(
            handles=[
                Patch(facecolor=BASELINE, label="Coconut / CODI baseline"),
                Patch(facecolor=SIMCOT, label="Baseline + SIM-CoT"),
            ],
            loc="upper left", bbox_to_anchor=(.304, .833),
            ncol=2, frameon=False, fontsize=13,
            handlelength=1.3, handleheight=.8, columnspacing=2.2,
        )
        fig.text(.922, .788, "Accuracy gain", ha="center", fontsize=12, color=MUTED)

        for i, row in enumerate(rows):
            y = len(rows) - 1 - i
            baseline = row["baseline_accuracy"]
            supervised = row["simcot_accuracy"]
            gain = Decimal(str(supervised)) - Decimal(str(baseline))

            ax.barh(y + .18, baseline, height=.28, color=BASELINE, zorder=2)
            ax.barh(y - .18, supervised, height=.28, color=SIMCOT, zorder=2)
            for position, value in ((y + .18, baseline), (y - .18, supervised)):
                ax.text(
                    value + 1.2, position, f"{value:.1f}",
                    va="center", ha="left", fontsize=15, color=INK,
                )

            label_transform = ax.get_yaxis_transform()
            ax.text(
                -.055, y + .10, row["backbone"], transform=label_transform,
                ha="right", va="center", fontsize=16, color=INK,
            )
            ax.text(
                -.055, y - .18, f'{row["method"]} · Table {row["table"]}',
                transform=label_transform, ha="right", va="center",
                fontsize=12.5, color=MUTED,
            )
            ax.text(
                1.20, y, f"+{gain:.1f} pp", transform=label_transform,
                va="center", ha="center", fontsize=18, color=INK,
            )

        ax.set_xlim(0, 100)
        ax.set_ylim(-.6, len(rows) - .4)
        ax.set_yticks([])
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="x", length=4, width=.8, pad=8, labelsize=12)
        ax.set_xlabel(source["results_metric"], labelpad=13, fontsize=13)
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(RULE)
        ax.spines["bottom"].set_linewidth(.8)

        # A simple paper-style note retains the comparison's original scope.
        fig.add_artist(Line2D([.055, .965], [.166, .166], transform=fig.transFigure,
                              color=RULE, linewidth=.8))
        fig.text(
            .055, .119,
            f'{efficiency["backbone"]} · {efficiency["implicit_method"]}: '
            f'{efficiency["implicit_accuracy"]:.1f}% accuracy with '
            f'{efficiency["implicit_tokens"]:.1f} average tokens',
            fontsize=14, color=INK,
        )
        fig.text(
            .055, .078,
            f'{efficiency["explicit_method"]}: {efficiency["explicit_accuracy"]:.1f}% '
            f'with {efficiency["explicit_tokens"]:.1f} average tokens · '
            f'{efficiency["explicit_tokens"]:.1f} / {efficiency["implicit_tokens"]:.1f} '
            f'= {efficiency["ratio_rounded"]:.1f}× token-count ratio (rounded)',
            fontsize=12.5, color=MUTED,
        )
        fig.text(
            .965, .032, f'Table {efficiency["table"]} · Average token counts',
            ha="right", fontsize=10.5, color=MUTED,
        )

        fig.savefig(OUT / "results-overview.png", facecolor=BACKGROUND,
                    metadata={"Title": "GSM8k-Aug results", "Source": source["paper"]})
        fig.savefig(OUT / "results-overview.svg", facecolor=BACKGROUND,
                    metadata={"Title": "GSM8k-Aug results", "Date": None,
                              "Description": "Matched comparisons from paper Tables 1–3."})
        plt.close(fig)


if __name__ == "__main__":
    results()
    for name in ("results-overview.png", "results-overview.svg"):
        print(f"{name}: {(OUT / name).stat().st_size:,} bytes")

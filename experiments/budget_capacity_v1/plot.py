"""Render saved diagnostic aggregates; no fitting, selection or data access."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / "docs/figures"
DATASETS = ("concrete_slump", "parkinsons", "air_quality")
TITLES = ("Concrete slump", "Parkinson", "Air Quality")
ARMS = ("joint8", "independent8", "joint_wide")
LABELS = ("Joint: 8 bases", "Independent: 8/output", "Joint: 8/output total")
COLORS = ("#0072B2", "#D55E00", "#009E73")
STYLES = ("-", "--", "-.")


def canvas(title):
    fig, axes = plt.subplots(2, 3, figsize=(7.480315, 5.196850), sharex=True)  # 190 x 132 mm
    fig.subplots_adjust(left=.09, right=.985, bottom=.125, top=.795, wspace=.34, hspace=.62)
    fig.suptitle(title, y=.98, fontsize=10)
    for i, ax in enumerate(axes.flat):
        ax.set_title(f'{"abcdef"[i]}   {TITLES[i % 3]}', loc="left", fontsize=8, pad=8)
        ax.grid(axis="y", color="#dddddd", linewidth=.4)
        ax.tick_params(labelsize=7)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("Shared-DAG complexity cap K", fontsize=7, labelpad=5)
    return fig, axes


def band(ax, x, mean, sd, color, style):
    mean, sd = np.asarray(mean), np.asarray(sd)
    ax.fill_between(x, mean-sd, mean+sd, step="post", color=color, alpha=.10, linewidth=0)
    ax.step(x, mean, where="post", color=color, linestyle=style, linewidth=1.15)


def export(fig, stem, qa_scripts):
    fig.canvas.draw()
    if qa_scripts:
        sys.path.insert(0, str(qa_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=DEST/f"{stem}.alignment.json",
            overlay_svg=DEST/f"{stem}.alignment.svg", tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(DEST/f"{stem}.svg")
    fig.savefig(DEST/f"{stem}.pdf")
    fig.savefig(DEST/f"{stem}.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-scripts", type=Path, help="directory containing audit_panel_alignment.py")
    args = parser.parse_args()
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["DejaVu Sans"],
        "font.size":7, "axes.labelsize":7, "xtick.labelsize":7, "ytick.labelsize":7,
        "legend.fontsize":7, "svg.fonttype":"none", "pdf.fonttype":42,
        "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.65})
    summary = json.loads((HERE/"analysis/summary.json").read_text(encoding="utf-8"))
    with (HERE/"analysis/paired.csv").open(encoding="utf-8", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert len(summary["default"]) == 18
    cap_grid = range(summary["max_cap"] + 1)
    DEST.mkdir(exist_ok=True)
    fig, axes = canvas("Search budget, basis capacity and prediction error")
    handles = [Line2D([], [], color=c, ls=s, label=label) for c, s, label in zip(COLORS, STYLES, LABELS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53,.935), ncol=3, frameon=False)
    for col, dataset in enumerate(DATASETS):
        for row, budget in enumerate((20, 60)):
            ax = axes[row, col]
            ax.set_title(f'{"abcdef"[row*3+col]}   {TITLES[col]} | {budget} CPU s', loc="left", fontsize=8, pad=8)
            for arm, color, style in zip(ARMS, COLORS, STYLES):
                cells = [r for r in summary["curves"] if (r["dataset"], r["arm"], r["budget"]) == (dataset, arm, budget)]
                assert [r["cap"] for r in cells] == list(cap_grid)
                mean = [r["metrics"]["outer_nrmse"]["mean"] if r["complete"] else np.nan for r in cells]
                sd = [r["metrics"]["outer_nrmse"]["sd"] if r["complete"] else np.nan for r in cells]
                band(ax, cap_grid, mean, sd, color, style)
            ax.set_xlim(0, summary["max_cap"])
        lo = min(ax.get_ylim()[0] for ax in axes[:, col])
        hi = max(ax.get_ylim()[1] for ax in axes[:, col])
        for ax in axes[:, col]:
            ax.set_ylim(lo, hi)
    for ax in axes[:, 0]:
        ax.set_ylabel("Outer-fold NRMSE (lower is better)", fontsize=7)
    fig.text(.5,.027,"Mean +/- 1 descriptive SD across 3 folds x 2 seeds; inner-validation-selected models.", ha="center", fontsize=7)
    export(fig, "budget_capacity_curves", args.qa_scripts)

    fig, axes = canvas("Paired changes separate budget and basis-capacity effects")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53,.935), ncol=3, frameon=False)
    for col, dataset in enumerate(DATASETS):
        for row in range(2):
            ax = axes[row, col]
            ax.axhline(0, color="#777777", ls=":", linewidth=.7)
            choices = [(f"{arm}:60-20", c, s) for arm, c, s in zip(ARMS, COLORS, STYLES)] if row == 0 else [
                ("joint_wide-joint8:20", "#666666", "--"), ("joint_wide-joint8:60", "#009E73", "-")]
            for comparison, color, style in choices:
                selected = [p for p in pairs if p["dataset"] == dataset and p["comparison"] == comparison]
                mean, sd = [], []
                for cap in cap_grid:
                    values = [float(p["delta_outer_nrmse"]) for p in selected if p["cap"] == str(cap) and p["status"] == "ok"]
                    assert len(values) <= 6
                    mean.append(statistics.mean(values) if len(values) == 6 else np.nan)
                    sd.append(statistics.pstdev(values) if len(values) == 6 else np.nan)
                band(ax, cap_grid, mean, sd, color, style)
            ax.set_xlim(0, summary["max_cap"])
            if row == 1:
                ax.set_title(f'{"abcdef"[row*3+col]}   {TITLES[col]} | wide - joint8', loc="left", fontsize=8, pad=8)
    axes[0,0].set_ylabel("NRMSE change: 60 s - 20 s", fontsize=7)
    axes[1,0].set_ylabel("NRMSE change: wide - joint8", fontsize=7)
    capacity_handles = [Line2D([], [], color=c, ls=s, label=label) for c, s, label in (
        ("#666666", "--", "Capacity effect at 20 s"), ("#009E73", "-", "Capacity effect at 60 s"))]
    fig.legend(handles=capacity_handles, loc="center", bbox_to_anchor=(.54,.49), ncol=2, frameon=False)
    fig.text(.5,.027,"Paired mean +/- 1 descriptive SD; negative favors more CPU (top) or more basis slots (bottom).", ha="center", fontsize=7)
    export(fig, "budget_capacity_paired", args.qa_scripts)
    print("Exported both figures. Alignment " + ("checked." if args.qa_scripts else "not checked; supply --qa-scripts for QA."))


if __name__ == "__main__":
    main()

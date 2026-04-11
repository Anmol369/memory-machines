"""
Visualize pluckability results: aggregate accuracy + per-tier breakdown.

Creates a 5-panel figure (Overall | T0 | T1 | T2 | T3) showing zero-shot
vs few-shot accuracy for each model.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
EXCLUDED_SOURCES_PATH = Path(__file__).parent / "instructions" / "few_shot_sources.json"

TASK_TYPE_TO_TIER = {
    "off-target": "T0",
    "needs-refactor": "T1",
    "needs-polish": "T2",
    "ready-to-review": "T3",
}

TIER_ORDER = ["T0", "T1", "T2", "T3"]

MODEL_DISPLAY = {
    "claude-opus-4-5": "Opus 4.5",
    "gpt-5.2": "GPT 5.2",
    "claude-sonnet-4-5": "Sonnet 4.5",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-oss-120b": "GPT OSS 120B",
    "qwen-3-32b": "Qwen3 32B",
}

# Colors
COLOR_ZERO = "#BEE9E8"
COLOR_FEW = "#1B4965"
BG_DEFAULT = "#FAFAFA"
BG_BOUNDARY = "#FFF3EE"
TEXT_COLOR = "#2C3E50"


def load_excluded_sources():
    if EXCLUDED_SOURCES_PATH.exists():
        with open(EXCLUDED_SOURCES_PATH) as f:
            return json.load(f).get("contaminated_sources", [])
    return []


def load_all_results(excluded_sources):
    rows = []
    for filepath in RESULTS_DIR.glob("results_*.jsonl"):
        parts = filepath.stem.replace("results_", "").split("_", 2)
        condition = parts[0] + "_" + parts[1]
        model = parts[2] if len(parts) > 2 else "unknown"

        with open(filepath) as f:
            for line in f:
                d = json.loads(line)
                if d.get("source_url") in excluded_sources:
                    continue
                if d.get("judge_prediction") is None:
                    continue
                rows.append({
                    "model": model,
                    "condition": condition,
                    "task_type": d.get("task_type"),
                    "correct": d.get("correct", False),
                })
    return rows


def compute_accuracies(rows):
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))

    for r in rows:
        model = r["model"]
        cond = r["condition"]
        tier = TASK_TYPE_TO_TIER.get(r["task_type"])

        counts["Overall"][model][cond][1] += 1
        if r["correct"]:
            counts["Overall"][model][cond][0] += 1

        if tier:
            counts[tier][model][cond][1] += 1
            if r["correct"]:
                counts[tier][model][cond][0] += 1

    result = {}
    for panel, models in counts.items():
        result[panel] = {}
        for model, conds in models.items():
            result[panel][model] = {}
            for cond, (corr, tot) in conds.items():
                result[panel][model][cond] = corr / tot if tot > 0 else 0
    return result


def compute_tier_counts(excluded_sources):
    tier_n = defaultdict(int)
    for filepath in sorted(RESULTS_DIR.glob("results_few_shot_claude-opus-4-5.jsonl")):
        with open(filepath) as f:
            for line in f:
                d = json.loads(line)
                if d.get("source_url") in excluded_sources:
                    continue
                tier = TASK_TYPE_TO_TIER.get(d.get("task_type"))
                if tier:
                    tier_n[tier] += 1
    return dict(tier_n)


def main():
    excluded = load_excluded_sources()
    rows = load_all_results(excluded)
    accs = compute_accuracies(rows)
    tier_n = compute_tier_counts(excluded)

    # Sort models by overall best accuracy (descending)
    models_sorted = sorted(
        MODEL_DISPLAY.keys(),
        key=lambda m: max(
            accs.get("Overall", {}).get(m, {}).get("few_shot", 0),
            accs.get("Overall", {}).get(m, {}).get("zero_shot", 0),
        ),
        reverse=True,
    )

    panels = ["Overall"] + TIER_ORDER
    panel_titles = {
        "Overall": "Overall",
        "T0": f"Off-target — T0\n(n={tier_n.get('T0', '?')})",
        "T1": f"Needs Refactor — T1\n(n={tier_n.get('T1', '?')})",
        "T2": f"Needs Polish — T2\n(n={tier_n.get('T2', '?')})",
        "T3": f"Ready to Review — T3\n(n={tier_n.get('T3', '?')})",
    }

    fig, axes = plt.subplots(
        1, 5, figsize=(16, 7.5), sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    width = 0.32
    x = np.arange(len(models_sorted))

    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        zero_accs = []
        few_accs = []
        for model in models_sorted:
            zero_accs.append(accs.get(panel, {}).get(model, {}).get("zero_shot", 0))
            few_accs.append(accs.get(panel, {}).get(model, {}).get("few_shot", 0))

        zero_accs = np.array(zero_accs)
        few_accs = np.array(few_accs)

        ax.bar(x - width / 2, zero_accs, width, color=COLOR_ZERO, alpha=0.9,
               edgecolor="white", linewidth=0.8)
        ax.bar(x + width / 2, few_accs, width, color=COLOR_FEW, alpha=0.9,
               edgecolor="white", linewidth=0.8)

        # 50% reference line
        ax.axhline(y=0.5, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=0)

        ax.set_title(panel_titles[panel], fontsize=9.5, fontweight="bold",
                      color=TEXT_COLOR, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY[m] for m in models_sorted],
            fontsize=8, rotation=45, ha="right",
        )
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.10))
        ax.grid(axis="y", alpha=0.12, linestyle="-", color="#B0B0B0")

        # Highlight T1/T2 panels
        if panel in ("T1", "T2"):
            ax.set_facecolor(BG_BOUNDARY)
        else:
            ax.set_facecolor(BG_DEFAULT)

        # y-axis formatting
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold", color=TEXT_COLOR)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.tick_params(axis="y", labelsize=8.5)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")

        # Dashed vertical separator after Overall panel
        if idx == 0:
            # Draw a subtle dashed line at the right edge
            bbox = ax.get_position()
            fig.add_artist(plt.Line2D(
                [bbox.x1 + 0.005, bbox.x1 + 0.005], [bbox.y0, bbox.y1],
                transform=fig.transFigure, color="#CCCCCC",
                linewidth=0.8, linestyle="--", clip_on=False,
            ))

    # Shared legend — top of first panel area
    legend_patches = [
        mpatches.Patch(color=COLOR_ZERO, label="Zero-shot", alpha=0.9),
        mpatches.Patch(color=COLOR_FEW, label="Few-shot", alpha=0.9),
    ]
    # Place legend inside the Overall panel (most headroom)
    axes[0].legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8.5,
        framealpha=0.95,
    )

    fig.subplots_adjust(bottom=0.22, top=0.88, left=0.045, right=0.995, wspace=0.06)

    output_path = RESULTS_DIR / "accuracy_by_tier.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "datasets",
#     "scikit-learn",
# ]
# ///
"""
Plot ROC curves for all trained pluckability classifiers.

Usage:
    uv run memory_machines/training/classifier/viz_roc_curves.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_curve, auc, precision_score, recall_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("memory_machines/training/classifier/results")

MODELS = {
    "qwen3-0.6b-final_test.json": "Qwen3-0.6B",
    "qwen3-14b-lora-final_test.json": "Qwen3-14B + LoRA",
}

TIER_MAP = {
    "off-target": "T0",
    "needs-refactor": "T1",
    "needs-polish": "T2",
    "ready-to-review": "T3",
}

# ---------------------------------------------------------------------------
# Load HuggingFace dataset for tier mapping
# ---------------------------------------------------------------------------
ds = load_dataset("laddermedia/srs-prompts")
id_to_tier = {}
for split in ds.values():
    for row in split:
        id_to_tier[row["id"]] = TIER_MAP[row["task_type"]]

print(f"Loaded tier mapping for {len(id_to_tier)} samples")

# ---------------------------------------------------------------------------
# Load model results
# ---------------------------------------------------------------------------
model_data = {}
for fname, label in MODELS.items():
    path = RESULTS_DIR / fname
    if not path.exists():
        print(f"Skipping {fname} (not found)")
        continue
    with open(path) as f:
        raw = json.load(f)
    ids = [row[0] for row in raw["data"]]
    y_true = np.array([row[1] for row in raw["data"]])
    prob_pluckable = np.array([row[4] for row in raw["data"]])
    model_data[label] = {
        "ids": ids,
        "y_true": y_true,
        "prob_pluckable": prob_pluckable,
    }

# ---------------------------------------------------------------------------
# Plot ROC curves
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

colors = ["#059669", "#9333ea"]

fig, ax = plt.subplots(figsize=(6, 6))

for (label, data), color in zip(model_data.items(), colors):
    fpr, tpr, _ = roc_curve(data["y_true"], data["prob_pluckable"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random baseline")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.legend(loc="lower right", fontsize=9, frameon=True, fancybox=False, edgecolor="#cccccc")
ax.set_aspect("equal")

fig.tight_layout()
out_path = RESULTS_DIR / "roc_curves.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nSaved ROC curves to {out_path}")
plt.close()

# ---------------------------------------------------------------------------
# Pluckability precision and per-tier breakdown
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Pluckability Metrics (threshold = 0.5)")
print("=" * 70)

for label, data in model_data.items():
    y_true = data["y_true"]
    y_pred = (data["prob_pluckable"] >= 0.5).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # FPR: fraction of unpluckable samples incorrectly predicted as pluckable
    neg_mask = y_true == 0
    fpr_val = y_pred[neg_mask].sum() / neg_mask.sum() if neg_mask.sum() > 0 else 0

    print(f"\n{label}")
    print(f"  Precision (pluckable): {prec:.3f}")
    print(f"  Recall (pluckable):    {rec:.3f}")
    print(f"  FPR (unpluckable predicted pluckable): {fpr_val:.3f}")

    # Per-tier breakdown
    ids = data["ids"]
    prob_pluck = data["prob_pluckable"]

    tier_stats: dict[str, dict[str, int]] = {}
    for i, sample_id in enumerate(ids):
        tier = id_to_tier.get(sample_id, "unknown")
        if tier not in tier_stats:
            tier_stats[tier] = {"correct": 0, "total": 0}
        tier_stats[tier]["total"] += 1

        pred_pluckable = prob_pluck[i] >= 0.5
        true_pluckable = y_true[i] == 1

        if pred_pluckable == true_pluckable:
            tier_stats[tier]["correct"] += 1

    print("  Per-tier accuracy:")
    for tier in ["T0", "T1", "T2", "T3"]:
        if tier in tier_stats:
            s = tier_stats[tier]
            pct = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            print(f"    {tier}: {s['correct']}/{s['total']} = {pct:.1f}%")

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Define results directory
RESULTS_DIR = Path(__file__).parent / "results"


def load_results(filepath, excluded_sources=None):
    """Load JSONL results file and calculate accuracy.

    Args:
        filepath: Path to the JSONL file
        excluded_sources: Optional list of source URLs to exclude

    Returns:
        Accuracy as a float between 0 and 1
    """
    correct_count = 0
    total_count = 0

    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)

            # Skip if source_url is in excluded list
            if excluded_sources and data.get("source_url") in excluded_sources:
                continue

            total_count += 1
            if data.get("correct", False):
                correct_count += 1

    return correct_count / total_count if total_count > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Visualize pluckability evaluation results as a bar chart.")
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="Include blocked/contaminated sources in visualization (by default they are excluded)",
    )
    parser.add_argument(
        "--blocked-sources-file",
        type=str,
        default="./instructions/few_shot_sources.json",
        help="Path to JSON file containing list of blocked source URLs (default: ./instructions/few_shot_sources.json)",
    )
    args = parser.parse_args()

    # Load blocked sources unless --include-blocked is set
    excluded_sources = None
    if not args.include_blocked:
        exclude_path = Path(args.blocked_sources_file)
        if not exclude_path.is_absolute():
            exclude_path = RESULTS_DIR.parent / exclude_path

        if exclude_path.exists():
            with open(exclude_path, "r") as f:
                exclude_data = json.load(f)
                excluded_sources = exclude_data.get("contaminated_sources", [])
            print(
                f"Excluding {len(excluded_sources)} blocked/contaminated sources (use --include-blocked to include them)"
            )
        else:
            print(f"Warning: Blocked sources file not found at {exclude_path}, proceeding without exclusions")
    else:
        print("Including blocked/contaminated sources in visualization")

    # Dictionary to store results: {model_name: {instruction_type: accuracy}}
    results = {}

    # Process all result files
    for filepath in RESULTS_DIR.glob("results_*.jsonl"):
        filename = filepath.stem  # e.g., "results_zero_shot_gpt-5.2"
        parts = filename.replace("results_", "").split("_", 2)

        # Extract instruction type and model name
        instruction_type = parts[0] + "_" + parts[1]  # e.g., "zero_shot" or "few_shot"
        model_name = parts[2] if len(parts) > 2 else "unknown"  # e.g., "gpt-5.2"

        # Load accuracy with excluded sources
        accuracy = load_results(filepath, excluded_sources)

        # Store in nested dictionary
        if model_name not in results:
            results[model_name] = {}
        results[model_name][instruction_type] = accuracy

        print(f"{model_name} ({instruction_type}): {accuracy:.2%}")

    # Prepare data for plotting - sort by maximum accuracy (descending)
    models = sorted(
        results.keys(),
        key=lambda m: max(results[m].get("zero_shot", 0), results[m].get("few_shot", 0)),
        reverse=True,
    )

    # Create data arrays
    zero_shot_accuracies = []
    few_shot_accuracies = []

    for model in models:
        zero_shot_accuracies.append(results[model].get("zero_shot", 0))
        few_shot_accuracies.append(results[model].get("few_shot", 0))

    # Set up the bar chart with default spacing
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    # Blue-tinted pastel color palette
    color_zero_shot = "#A8DADC"  # Light blue pastel
    color_few_shot = "#457B9D"  # Deeper blue pastel

    # Create bars with rounded corners
    def create_rounded_bars(x_positions, heights, width, color, label):
        bars = []
        for i, (x_pos, height) in enumerate(zip(x_positions, heights)):
            # Create a rounded rectangle patch
            rounded_rect = mpatches.FancyBboxPatch(
                (x_pos - width / 2, 0),
                width,
                height,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                edgecolor="white",
                facecolor=color,
                linewidth=1.5,
                alpha=0.9,
            )
            ax.add_patch(rounded_rect)
            bars.append(rounded_rect)
        return bars

    create_rounded_bars(x - width / 2, zero_shot_accuracies, width, color_zero_shot, "Zero-shot")
    create_rounded_bars(x + width / 2, few_shot_accuracies, width, color_few_shot, "Few-shot")

    # Add value labels above bars
    def add_value_labels(x_positions, heights, offset):
        for x_pos, height in zip(x_positions, heights):
            if height > 0:  # Only show label if there's data
                ax.text(
                    x_pos + offset,
                    height + 0.02,  # Small padding above the bar
                    f"{height:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#2C3E50",
                )

    add_value_labels(x, zero_shot_accuracies, -width / 2)
    add_value_labels(x, few_shot_accuracies, width / 2)

    # Customize the plot
    ax.set_xlabel("Model", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_title(
        "Model Accuracy: Zero-shot vs Few-shot", fontsize=15, fontweight="bold", pad=20, color="#2C3E50"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)

    # Create custom legend with rounded patches at the top right
    legend_patches = [
        mpatches.Patch(color=color_zero_shot, label="Zero-shot", alpha=0.9),
        mpatches.Patch(color=color_few_shot, label="Few-shot", alpha=0.9),
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=11,
        loc="upper right",
        framealpha=0.95,
    )

    # Set axis limits to ensure all models fit
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#B0B0B0")
    ax.set_facecolor("#FAFAFA")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_path = RESULTS_DIR / "accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()

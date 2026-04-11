import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

from datasets import Dataset, load_dataset
from memory_machines.utils.tiers import get_tier_from_task_type

# Define results directory
RESULTS_DIR = Path(__file__).parent / "results"

# Color scheme for tiers (gradient representing quality: green → amber → coral)
TIER_COLORS = {
    'T3': '#2A9D8F',  # Teal/turquoise (best quality - ready to review)
    'T2': '#E9C46A',  # Warm amber/gold (medium quality - needs polish)
    'T1': '#F4A261',  # Coral/salmon (needs work - needs refactor)
}


def load_results(filepath, highlight_to_source=None, excluded_sources=None):
    """Load JSONL results file and calculate tier distribution.

    Args:
        filepath: Path to the JSONL file
        highlight_to_source: Dictionary mapping highlight_id to source_url
        excluded_sources: Optional list of source URLs to exclude

    Returns:
        Dictionary with tier percentages: {'T3': float, 'T2': float, 'T1': float}
    """
    tier_counts = {'T3': 0, 'T2': 0, 'T1': 0}
    total_count = 0

    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)

            # Skip if source_url is in excluded list (map highlight_id to source_url)
            if excluded_sources and highlight_to_source:
                highlight_id = data.get("highlight_id")
                if highlight_id is not None:
                    source_url = highlight_to_source.get(highlight_id)
                    if source_url in excluded_sources:
                        continue

            # Get chosen task type and map to tier
            chosen_task_type = data.get("chosen_task_type")
            if chosen_task_type is None:
                continue

            try:
                tier = get_tier_from_task_type(chosen_task_type)

                # Only count T1, T2, T3 (exclude T0/off-target)
                if tier in ['T1', 'T2', 'T3']:
                    tier_counts[tier] += 1
                    total_count += 1
            except (KeyError, ValueError):
                # Skip invalid task types
                continue

    # Calculate percentages
    if total_count > 0:
        tier_percentages = {
            tier: (count / total_count) * 100
            for tier, count in tier_counts.items()
        }
    else:
        tier_percentages = {'T3': 0, 'T2': 0, 'T1': 0}

    return tier_percentages


def lighten_color(hex_color, amount=0.5):
    """Lighten a hex color by blending with white.

    Args:
        hex_color: Hex color string (e.g., '#2A9D8F')
        amount: Amount to lighten (0-1, where 1 is white)

    Returns:
        Lightened hex color string
    """
    # Remove '#' and convert to RGB
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

    # Blend with white
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)

    return f'#{r:02x}{g:02x}{b:02x}'


def create_stacked_rounded_bars(ax, x_pos, tier_heights, width, colors, use_hatch=False):
    """Create stacked bar with rounded corners using FancyBboxPatch.

    Args:
        ax: Matplotlib axes object
        x_pos: X position for the center of the bar
        tier_heights: Dictionary with tier percentages: {'T3': float, 'T2': float, 'T1': float}
        width: Width of the bar
        colors: Dictionary mapping tiers to colors
        use_hatch: If True, apply dotted pattern with lighter color to the bar
    """
    cumulative_height = 0

    # Create segments from bottom to top: T3 → T2 → T1
    for tier in ['T3', 'T2', 'T1']:
        height = tier_heights[tier]

        # Skip if height is zero
        if height > 0:
            # Use lighter color for hatch pattern, white for solid bars
            if use_hatch:
                edge_color = lighten_color(colors[tier], amount=0.6)
            else:
                edge_color = "white"

            rounded_rect = mpatches.FancyBboxPatch(
                (x_pos - width / 2, cumulative_height),
                width,
                height,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                edgecolor=edge_color,
                facecolor=colors[tier],
                linewidth=1.5,
                alpha=0.9,
                hatch='...' if use_hatch else None,
            )
            ax.add_patch(rounded_rect)

        cumulative_height += height


def add_segment_labels(ax, x_pos, tier_heights, offset):
    """Add percentage labels to each segment of stacked bar.

    Args:
        ax: Matplotlib axes object
        x_pos: X position of the model group
        tier_heights: Dictionary with tier percentages
        offset: X offset for instruction type (-width/2 or +width/2)
    """
    cumulative = 0

    # Add labels for each tier (bottom to top)
    for tier in ['T3', 'T2', 'T1']:
        height = tier_heights[tier]

        # Only show label if segment is large enough (>5%)
        if height > 5:
            y_pos = cumulative + height / 2  # Center of segment
            ax.text(
                x_pos + offset, y_pos,
                f"{height:.1f}%",
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='#2C3E50'
            )

        cumulative += height


def main():
    parser = argparse.ArgumentParser(
        description="Visualize forced-choice tier distribution as stacked bar chart."
    )
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="Include blocked/contaminated sources in visualization (by default they are excluded)",
    )
    parser.add_argument(
        "--blocked-sources-file",
        type=str,
        default="./instructions/few_shot_sources.json",
        help="Path to JSON file containing list of blocked source URLs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filepath (default: results/tier_distribution_stacked.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively (default: save only)",
    )
    args = parser.parse_args()

    # Load dataset and blocked sources unless --include-blocked is set
    excluded_sources = None
    highlight_to_source = None

    if not args.include_blocked:
        # Load dataset to map highlight_id to source_url
        print("Loading dataset from laddermedia/srs-highlights")
        try:
            dataset = load_dataset("laddermedia/srs-highlights", split="grounded")
            assert isinstance(dataset, Dataset), "Dataset is not a Dataset"

            # Create mapping from highlight_id to source_url
            highlight_to_source = {}
            for highlight in dataset:
                highlight_to_source[highlight["highlight_id"]] = highlight["source_url"]
        except Exception as e:
            print(f"Warning: Failed to load dataset: {e}")
            print("Proceeding without source filtering")

        # Load blocked sources
        if highlight_to_source is not None:
            exclude_path = Path(args.blocked_sources_file)
            if not exclude_path.is_absolute():
                exclude_path = RESULTS_DIR.parent / exclude_path

            if exclude_path.exists():
                with open(exclude_path, "r") as f:
                    exclude_data = json.load(f)
                    excluded_sources = exclude_data.get("contaminated_sources", [])
                print(
                    f"Excluding {len(excluded_sources)} blocked/contaminated sources "
                    f"(use --include-blocked to include them)"
                )
            else:
                print(f"Warning: Blocked sources file not found at {exclude_path}")
    else:
        print("Including blocked/contaminated sources in visualization")

    # Dictionary to store results: {model_name: {instruction_type: tier_distribution}}
    results = {}

    # Process all result files
    for filepath in RESULTS_DIR.glob("*_reasoning_none.jsonl"):
        filename = filepath.stem  # e.g., "simple_claude-sonnet-4-5_reasoning_none"

        # Remove the _reasoning_none suffix
        filename = filename.replace("_reasoning_none", "")

        # Split to extract instruction type and model
        parts = filename.split("_", 1)
        if len(parts) < 2:
            continue

        instruction_type = parts[0]  # e.g., "simple" or "few_shot" (few + shot)
        model_name = parts[1]  # e.g., "claude-sonnet-4-5"

        # Handle few_shot case
        if instruction_type == "few":
            remaining = parts[1].split("_", 1)
            if len(remaining) >= 2 and remaining[0] == "shot":
                instruction_type = "few_shot"
                model_name = remaining[1]
            else:
                continue

        # Load tier distribution
        tier_distribution = load_results(filepath, highlight_to_source, excluded_sources)

        # Validate percentages sum to approximately 100%
        total = sum(tier_distribution.values())
        if total > 0 and not (99.0 <= total <= 101.0):
            print(f"Warning: {filename} percentages sum to {total:.1f}%")

        # Store in nested dictionary
        if model_name not in results:
            results[model_name] = {}
        results[model_name][instruction_type] = tier_distribution

        print(f"{model_name} ({instruction_type}): T3={tier_distribution['T3']:.1f}%, "
              f"T2={tier_distribution['T2']:.1f}%, T1={tier_distribution['T1']:.1f}%")

    if not results:
        print("Error: No result files found")
        return

    # Prepare data for plotting
    models = sorted(results.keys())

    # Set up the bar chart
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create stacked bars for each model and instruction type
    for i, model in enumerate(models):
        # Simple instruction (left bar - solid)
        if 'simple' in results[model]:
            simple_dist = results[model]['simple']
            create_stacked_rounded_bars(
                ax, x[i] - width / 2, simple_dist, width, TIER_COLORS, use_hatch=False
            )
            add_segment_labels(ax, x[i], simple_dist, -width / 2)

        # Few-shot instruction (right bar - cross-hatched)
        if 'few_shot' in results[model]:
            few_shot_dist = results[model]['few_shot']
            create_stacked_rounded_bars(
                ax, x[i] + width / 2, few_shot_dist, width, TIER_COLORS, use_hatch=True
            )
            add_segment_labels(ax, x[i], few_shot_dist, width / 2)

    # Customize the plot
    ax.set_xlabel("Model", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_ylabel("Distribution (%)", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_title(
        "Tier Distribution: Simple vs Few-shot Instructions",
        fontsize=15,
        fontweight="bold",
        pad=20,
        color="#2C3E50"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)

    # Create legend
    legend_patches = [
        mpatches.Patch(color=TIER_COLORS['T3'], label='T3 (Ready to Review)', alpha=0.9),
        mpatches.Patch(color=TIER_COLORS['T2'], label='T2 (Needs Polish)', alpha=0.9),
        mpatches.Patch(color=TIER_COLORS['T1'], label='T1 (Needs Refactor)', alpha=0.9),
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=11,
        loc="upper right",
        framealpha=0.95,
    )

    # Add instruction type annotation
    fig.text(
        0.5, 0.02,
        "Solid bars: Simple instruction  |  Dotted bars (...): Few-shot instruction",
        ha='center',
        fontsize=10,
        color='#2C3E50'
    )

    # Set axis limits and styling
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#B0B0B0")
    ax.set_facecolor("#FAFAFA")

    # Adjust layout to make room for annotation
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save the figure
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_DIR / "tier_distribution_stacked.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"\nChart saved to: {output_path}")

    # Show the plot if requested
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

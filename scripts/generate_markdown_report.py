#!/usr/bin/env python3
"""
Generate a comprehensive markdown report from SPEAR model results.
"""

from pathlib import Path
import pandas as pd
import sys


def generate_markdown_report(summary_csv: Path, output_md: Path):
    """Generate a markdown report from the summary CSV."""
    df = pd.read_csv(summary_csv)

    # Filter to completed runs with metrics
    completed = df[df["has_metrics"]].copy()

    with open(output_md, "w") as f:
        f.write("# SPEAR Model Results Summary\n\n")
        f.write(f"Generated from {len(df)} total model runs\n\n")
        f.write(f"- Completed runs with metrics: **{len(completed)}**\n")
        f.write(f"- Still running: **{len(df) - len(completed)}**\n\n")

        f.write("---\n\n")

        # Overall statistics
        f.write("## Overall Statistics\n\n")

        for dataset in sorted(completed["dataset"].dropna().unique()):
            dataset_df = completed[completed["dataset"] == dataset]
            f.write(f"### {dataset.capitalize()} Dataset\n\n")

            for gene_count in sorted(dataset_df["gene_count"].dropna().unique()):
                gene_df = dataset_df[dataset_df["gene_count"] == gene_count]

                if len(gene_df) == 0:
                    continue

                f.write(f"#### {gene_count} Genes ({len(gene_df)} models)\n\n")

                # Create a table
                f.write(
                    "| Rank | Model | Test Pearson | Test Spearman | Test R² | Test RMSE |\n"
                )
                f.write(
                    "|------|-------|--------------|---------------|---------|--------|\n"
                )

                # Sort by test Pearson
                ranked = gene_df.sort_values("test_pearson_mean", ascending=False)

                for rank, (idx, row) in enumerate(ranked.iterrows(), 1):
                    model = row["model_id"]
                    pearson = row.get("test_pearson_mean", None)
                    spearman = row.get("test_spearman_mean", None)
                    r2 = row.get("test_r2_mean", None)
                    rmse = row.get("test_rmse_mean", None)

                    pearson_std = row.get("test_pearson_std", None)

                    # Format with std if available
                    if (
                        pearson is not None
                        and pearson_std is not None
                        and pd.notna(pearson)
                        and pd.notna(pearson_std)
                    ):
                        pearson_str = f"{pearson:.4f} ± {pearson_std:.4f}"
                    elif pearson is not None and pd.notna(pearson):
                        pearson_str = f"{pearson:.4f}"
                    else:
                        pearson_str = "N/A"

                    spearman_str = (
                        f"{spearman:.4f}"
                        if spearman is not None and pd.notna(spearman)
                        else "N/A"
                    )
                    r2_str = f"{r2:.4f}" if r2 is not None and pd.notna(r2) else "N/A"
                    rmse_str = (
                        f"{rmse:.4f}" if rmse is not None and pd.notna(rmse) else "N/A"
                    )

                    f.write(
                        f"| {rank} | {model} | {pearson_str} | {spearman_str} | {r2_str} | {rmse_str} |\n"
                    )

                f.write("\n")

                # Add key findings
                top3 = ranked.head(3)
                f.write("**Top 3 Models:**\n\n")
                for rank, (idx, row) in enumerate(top3.iterrows(), 1):
                    pearson = row.get("test_pearson_mean", None)
                    if pearson is not None and pd.notna(pearson):
                        pearson_str = f"{pearson:.4f}"
                    else:
                        pearson_str = "N/A"
                    f.write(
                        f"{rank}. **{row['model_id']}**: {pearson_str} mean test Pearson correlation\n"
                    )
                f.write("\n")

        f.write("---\n\n")

        # Model comparison across datasets
        f.write("## Model Comparison Across Datasets\n\n")

        if len(completed["dataset"].dropna().unique()) > 1:
            # Find models that ran on both datasets
            models_by_dataset = {}
            for dataset in completed["dataset"].dropna().unique():
                models_by_dataset[dataset] = set(
                    completed[completed["dataset"] == dataset]["model_id"]
                )

            common_models = (
                set.intersection(*models_by_dataset.values())
                if models_by_dataset
                else set()
            )

            if common_models:
                f.write(f"Models tested on multiple datasets: {len(common_models)}\n\n")

                f.write(
                    "| Model | Embryonic Pearson | Endothelial Pearson | Difference |\n"
                )
                f.write(
                    "|-------|-------------------|---------------------|------------|\n"
                )

                for model in sorted(common_models):
                    emb_row = completed[
                        (completed["dataset"] == "embryonic")
                        & (completed["model_id"] == model)
                    ]
                    end_row = completed[
                        (completed["dataset"] == "endothelial")
                        & (completed["model_id"] == model)
                    ]

                    if len(emb_row) > 0 and len(end_row) > 0:
                        emb_pearson = emb_row.iloc[0]["test_pearson_mean"]
                        end_pearson = end_row.iloc[0]["test_pearson_mean"]
                        diff = emb_pearson - end_pearson

                        f.write(
                            f"| {model} | {emb_pearson:.4f} | {end_pearson:.4f} | {diff:+.4f} |\n"
                        )

                f.write("\n")

        # Status of incomplete runs
        incomplete = df[~df["has_metrics"]]
        if len(incomplete) > 0:
            f.write("---\n\n")
            f.write("## Incomplete Runs\n\n")
            f.write(
                f"The following {len(incomplete)} runs are still in progress or failed:\n\n"
            )

            for dataset in sorted(incomplete["dataset"].dropna().unique()):
                dataset_incomplete = incomplete[incomplete["dataset"] == dataset]
                f.write(f"### {dataset.capitalize()}\n\n")
                for idx, row in dataset_incomplete.iterrows():
                    f.write(f"- {row['model_id']} ({row['run_name']})\n")
                f.write("\n")

    print(f"Markdown report saved to: {output_md}")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    summary_csv = (
        project_root / "analysis" / "reports" / "summary_metrics_all_models.csv"
    )
    output_md = project_root / "analysis" / "reports" / "RESULTS_SUMMARY.md"

    if not summary_csv.exists():
        print(f"Error: Summary CSV not found: {summary_csv}", file=sys.stderr)
        print("Run generate_summary_statistics.py first.", file=sys.stderr)
        sys.exit(1)

    generate_markdown_report(summary_csv, output_md)
    print(f"\nView the report at: {output_md}")


if __name__ == "__main__":
    main()

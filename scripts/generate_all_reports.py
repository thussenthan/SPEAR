#!/usr/bin/env python3
"""
Master script to generate all summary reports for SPEAR model runs.
This combines summary statistics generation and markdown reporting.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run all report generation scripts."""
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"

    print("=" * 80)
    print("SPEAR RESULTS SUMMARY GENERATOR")
    print("=" * 80)
    print()

    # Step 1: Generate summary statistics CSV
    print("Step 1: Generating summary statistics CSV...")
    print("-" * 80)
    try:
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "generate_summary_statistics.py")],
            cwd=project_root,
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"Error generating summary statistics (exit code {result.returncode})"
            )
            return 1
    except Exception as e:
        print(f"Failed to run generate_summary_statistics.py: {e}")
        return 1

    print()

    # Step 2: Generate markdown report
    print("Step 2: Generating markdown report...")
    print("-" * 80)
    try:
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "generate_markdown_report.py")],
            cwd=project_root,
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error generating markdown report (exit code {result.returncode})")
            return 1
    except Exception as e:
        print(f"Failed to run generate_markdown_report.py: {e}")
        return 1

    print()
    print("=" * 80)
    print("✓ All reports generated successfully!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - CSV: analysis/reports/summary_metrics_all_models.csv")
    print("  - MD:  analysis/reports/RESULTS_SUMMARY.md")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

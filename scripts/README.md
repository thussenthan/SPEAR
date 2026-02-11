# Utility Scripts

## Overview

Summary of helper scripts for data preparation, aggregation, and reporting.

## Inputs

- Repository checkout with `scripts/`.
- Data files referenced by individual scripts (see script docstrings or `--help`).

## Outputs

- Processed datasets under `data/`.
- Reports under `analysis/reports/`.
- Figures under `analysis/figs/`.

## Usage

### Core Pipeline Scripts

- `select_random_genes.py` – validate expression thresholds and build reusable gene manifests.
- `combine_filtered.py`, `preprocess.py` – data wrangling helpers used prior to training
- `preprocess_endothelial.py` – preprocess endothelial dataset
- `download_mesc_raw_data.py` – download raw mouse ESC data
- `combine_chunk_results.py` – stitch together per-chunk training outputs into unified result folders.
- `plot_feature_importance_vs_tss.py` – plot feature importance relative to TSS
- `preflight_check.py` – pre-flight checks before running pipeline

### Results Analysis & Reporting Scripts

**NEW:** Automated reporting tools for analyzing completed model runs:

- **`generate_all_reports.py`** – **[MAIN]** Generate all summary reports in one command
- **`generate_summary_statistics.py`** – Generate detailed CSV summary of all model runs
- **`generate_markdown_report.py`** – Generate human-readable markdown report

### Quick Start for Results Summary

After your model runs complete (or while they're running), generate a comprehensive summary:

```bash
# Generate both CSV and markdown reports
python scripts/generate_all_reports.py
```

This creates:

- `analysis/reports/summary_metrics_all_models.csv` - Detailed metrics table with performance stats
- `analysis/reports/RESULTS_SUMMARY.md` - Human-readable markdown report with rankings

### Jupyter Notebook Analysis

For interactive visualization and figure generation, use:

- `analysis/manuscript_figures.ipynb` – Interactive analysis and manuscript figure generation

The notebook expects results in `output/results/` and will generate publication-ready figures.

Run any script with `python scripts/<name>.py --help` for available options.

## References

- `README.md`

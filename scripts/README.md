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

- `download_geo_raw_data.py` – canonical raw-data download/staging entrypoint for embryonic and endothelial datasets.
- `preprocess_geo_raw_to_spear.py` – canonical sample-first preprocessing entrypoint for GEO raw multiome inputs.
- `generate_all_sample_manifests.py` – build per-sample `random`, `hvg`, and `low_noisy` manifests, with optional peak-count annotation/filtering for peak-only sweeps.
- `prepare_10x_pbmc_multiome.py` – prepare the public 10x PBMC multiome matrix into paired SPEAR-compatible RNA/ATAC AnnData files and manifests.
- `create_tiny_example_data.py` – create a synthetic paired ATAC/RNA mini dataset for install and CLI smoke tests.
- `list_data_samples.py` – emit canonical sample order for Slurm arrays and wrappers.
- `list_duplicate_runs.py` – inspect W&B projects for duplicate run names or fingerprints.
- `combine_chunk_results.py` – stitch together per-chunk training outputs into unified result folders.
- `plot_feature_importance_vs_tss.py` – plot feature importance relative to TSS.
- `plot_shap_vs_tss.py` – plot SHAP attribution relative to TSS.
- `plot_prediction_structure.py` – compare real versus predicted RNA cell structure from exported raw predictions.
- `preflight_check.py` – pre-flight checks before running the training pipeline.
- `preflight_wandb_sweeps.py` – validate local sweep YAML paths, manifest row counts, W&B tags/groups, model names, cache directories, and expected run-count accounting before W&B submission.

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

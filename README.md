# SPEAR: **S**ingle-cell-based **P**rediction of Gene **E**xpression from Chromatin **A**ccessibility **R**eadouts

![SPEAR Logo](docs/images/spear_logo.png)

## Overview

SPEAR (Single-cell-based Prediction of Gene Expression from Chromatin Accessibility Readouts) is an end-to-end machine-learning framework for training and evaluating multi-omics integration models. The toolkit focuses on predicting the full RNA gene expression vector for every cell directly from its paired single-cell ATAC accessibility profile, pairing reproducible preprocessing with a modular model zoo, batchable CLI entry points, and plotting notebooks for downstream analysis.

## Inputs

- Paired ATAC/RNA `h5ad` files with overlapping barcodes.
- A reference GTF containing gene annotations.
- Gene manifests or explicit gene lists.
- Optional JSON configuration via `--config-json`.

## Outputs

- Run artifacts under `output/results/` (metrics, predictions, histories, model files).
- Logs under `output/logs` (pipeline logs: `<run_name>.log`; Slurm logs: `spear_<jobid>_<task>.out/.err`).
- Figures under `analysis/figs`.

## Usage

### SPEAR Framework Overview

![SPEAR framework figure](docs/images/SPEAR%20Framework%20Figure_transparent%20background.png)

### Key Features

### Project Goals

- Predict the full RNA expression vector for every cell directly from its paired single-cell ATAC accessibility profile.
- Build feature matrices using ±10 kb windows around each gene's transcription start site, binned at 500 bp (40 bins per gene) and recomputed for every run.
- Provide a modular model zoo spanning convolutional, recurrent, transformer, graph-based, deep & cross networks, gradient boosting (XGBoost, CatBoost), multilayer perceptron, tree ensembles, and linear baselines (Ridge, Elastic Net, Lasso, OLS) so comparative experiments are one CLI flag away.
- Produce test-set diagnostics (scatter plots, per-gene Pearson summaries, epoch histories) while persisting raw predictions for reproducibility and downstream analysis.

- **Modal-specific normalization:**
  - ATAC/RNA matrices are subset to shared barcodes and aligned to a common ordering; by default ATAC uses TF–IDF (`tfidf` layer), with alternatives (`counts_per_million`, `log1p_cpm`, or none) available via the `atac_layer` setting.
  - RNA counts are converted to counts-per-million and log-transformed (`log1p_cpm` layer) if needed; double log1p transforms are skipped when a normalized layer already exists.
  - Optional `StandardScaler`/`MinMaxScaler` may be applied on features and targets (target scaling is skipped for log-transformed targets unless `force_target_scaling=True`).
- **k-NN smoothing:** Each cell is smoothed by averaging with its k nearest neighbors (default k=19 for `smoothing_k=20`) using PCA-informed nearest-neighbor search to reduce sparsity while maintaining dataset size.
- **Optional pseudobulk aggregation:** PCA-informed, group-aware pooling within each sample when `pseudobulk_group_size > 1`.
- **Group-aware splitting:** 70/15/15 train/val/test splits with `GroupShuffleSplit` keyed by `group_key` (default `sample`; falls back to random when insufficient groups), plus 5-fold cross-validation using `GroupKFold` when possible.
- **Model zoo:** CNN, ResNet, RNN, LSTM, Transformer, Graph (implicit message passing), DCN (Deep & Cross Network), PyTorch MLP, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, CatBoost, SVR, Ridge, Elastic Net, Lasso, and OLS. Each model is defined in `spear.models` and accessible through the CLI.
- **Unified diagnostics:** `analysis/spear_results_analysis.ipynb` replaces prior plotting scripts, generating per-gene Pearson summaries, violin plots, top-genes scatter plots, RMSE comparisons, prediction-vs-truth charts, and epoch history curves directly from run outputs.

### Datasets

- Mouse embryonic multiome (GSE205117): `docs/mouse_esc_dataset.md`
- Human hemogenic endothelium multiome (GSE270141): `docs/endothelial_dataset.md`

### Repository Layout

```text
analysis/figs/               # Notebook outputs and generated figures
analysis/spear_results_analysis.ipynb
data/                        # Local AnnData matrices, manifests, references (not published)
src/                         # Core Python package (config, data, training, evaluation)
scripts/                     # Minimal helper scripts (data prep only)
todo.md                      # Running task list
```

### Installation

1. Create/activate your environment (Python ≥ 3.10).
2. Install the Python requirements and the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

> Torch and XGBoost wheels can be large on HPC systems—consider using `pip install --no-cache-dir` if disk quotas are tight.
> Data files are not published with this repository; fetch or generate them locally before running the pipeline.

### Data Requirements

- Paired ATAC/RNA `h5ad` files with overlapping barcodes.
- A reference GTF containing gene annotations.
- See `docs/mouse_esc_dataset.md` and `docs/endothelial_dataset.md` for dataset-specific provenance and storage conventions.

### Running the Pipeline

The CLI exposes all data preparation and model training settings. Basic example (either `spear` or `python -m spear.cli`):

```bash
spear \
  --base-dir "$(pwd)" \
  --models lstm transformer \
  --gene-manifest /path/to/selected_genes.csv \
  --epochs 100 \
  --pseudobulk-group-size 20 \
  --device auto
```

SPEAR can be run on Slurm-managed clusters; job scripts are internal.

More flags and defaults are documented in `docs/config_reference.md`.

Environment / CLI highlights:

- `--gene-manifest` guarantees that every model trains on the same gene subset.
- `--cache-dir` enables on-disk reuse of preprocessing (recommended for repeated model runs).
- `--chromosomes genome-wide` explicitly disables chromosome filtering; provide a list to restrict loci.
- `--run-name` customises the output directory name (default: `spear_<model>_<genes>_<dataset>_<cpu/gpu>_<timestamp>`; missing components are omitted).
- W&B run name defaults to `<model>_<genes>_<dataset>` unless `--wandb-run-name` is provided.
- `--device` supports `cuda`, `cpu`, or `auto` (prefers CUDA when available; falls back otherwise).
- `--disable-pseudobulk` is a quick toggle to benchmark true single-cell training (equivalent to setting `--pseudobulk-group-size 1`).
- `--atac-layer` lets you swap CPM for alternative ATAC transforms such as `tfidf` or disable normalisation entirely.

### Generate Gene Manifests

Generate the 1000-gene manifests used in this repo with the helper script:

```bash
PYTHONPATH=src python scripts/select_random_genes.py \
  --base-dir /gpfs/Home/tqw5435/uzun/spear \
  --gene-count 1000 \
  --gtf-path /gpfs/Home/tqw5435/uzun/spear/data/references/GCF_000001635.27_genomic.gtf \
  --output /gpfs/Home/tqw5435/uzun/spear/data/embryonic/manifests/1000_random_genes.csv

PYTHONPATH=src python scripts/select_random_genes.py \
  --base-dir /gpfs/Home/tqw5435/uzun/spear \
  --gene-count 1000 \
  --rna-path '/gpfs/Home/tqw5435/uzun/spear/data/endothelial/processed/combined_RNA_qc_<15%mito.h5ad' \
  --gtf-path /gpfs/Home/tqw5435/uzun/spear/data/references/gencode.v44.annotation.gtf.gz \
  --output /gpfs/Home/tqw5435/uzun/spear/data/endothelial/manifests/1000_random_genes.csv
```

### Example runs

```bash
# Per-gene baselines on CPU
spear --per-gene --models ridge lasso ols --device cpu --run-name per_gene_baselines

# Multi-output torch run with smaller smoothing and no pseudobulk
spear --models mlp transformer --smoothing-k 5 --disable-pseudobulk --run-name multi_output_no_bulk

Optional experiment tracking is available via Weights & Biases. Enable it with `--wandb` after installing `wandb`
and exporting `WANDB_API_KEY` (or configuring `~/.netrc`). If the key or login is missing, SPEAR will skip W&B
logging without failing the run.

Quick setup:
1. Install: `pip install wandb`
2. Export your API key: `export WANDB_API_KEY=...`
3. Run with W&B enabled: `spear ... --wandb --wandb-project SPEAR`

By default, SPEAR logs summary metrics, tables, and key plots to W&B (with row/media caps). Use
`--wandb-no-artifacts`, `--wandb-no-tables`, `--wandb-no-media`, or `--wandb-no-predictions-table`
to trim logging.
```

### Results & Visualization

1. Run the CLI to generate metrics, predictions, histories, and fitted artifacts.
2. Open `analysis/spear_results_analysis.ipynb` and execute the cells. Adjust `RUN_INCLUDE_GLOBS` at the top of the notebook if you want to focus on specific runs.
3. Generated figures (violin plots, RMSE bars, scatter plots, epoch curves) are written back to `analysis/figs/` and CSV summaries are stored alongside them.

Only per-gene **test-set** Pearson correlations are emphasised in the visualizations. Validation metrics remain available for context (e.g., epoch curves) but are not part of the main comparisons.

### Output Layout

Run artifacts are written under `output/results/<run_name>/` with subfolders for each model. Each model directory includes metrics, predictions, histories, and (when enabled) feature-importance/SHAP exports.

### Feature Importance & SHAP Artifacts

Multi-output torch runs can emit feature-importance and SHAP summaries under each model directory when enabled via `--enable-feature-importance` and `--enable-shap`. Non-torch models do not produce SHAP outputs in the current pipeline.

- `feature_importances_mean.csv`, `feature_importances_raw.npz`, `feature_importance_per_gene_summary.csv`
- `feature_importance_mean.png`, `feature_importance_vs_tss_distance.png`
- `shap_importances_mean.csv`, `shap_importance_mean.png`

Use `scripts/plot_feature_importance_vs_tss.py` to build a publication-ready panel from these outputs.

### Supported Models

The following identifiers can be supplied to `--models` (and combined arbitrarily):

- `cnn`
- `resnet`
- `rnn`
- `lstm`
- `transformer`
- `graph`
- `dcn`
- `mlp`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`
- `xgboost`
- `catboost`
- `ridge`
- `elastic_net`
- `lasso`
- `ols`
- `svr`

Each mapping to a display label is tracked in `scripts/model_name_lookup.tsv` and consumed by the plotting notebook.
SVR defaults to a linear kernel with configurable hyperparameters via `TrainingConfig.svr_*`.

### Utilities and scripts

- `scripts/preflight_check.py`: quick readiness probe (env, packages, data paths, GTF). Run `python scripts/preflight_check.py --help` for options.
- `scripts/preprocess_endothelial.py`: RDS→AnnData conversion and QC for the endothelial dataset (barcode alignment, MT filtering, min genes/cells).

### Dependencies

All dependencies (runtime, dev, and notebooks) are listed in `requirements.txt`.

### Preprocessing Details

1. **AnnData loading:** ATAC and RNA `h5ad` matrices are loaded through `anndata`, subset to shared barcodes, and aligned to a common ordering.
2. **Modal layers:**
   - ATAC: TF–IDF is created by default (`training.atac_layer='tfidf'`); alternative transforms (such as CPM) can be requested via configuration.
   - RNA: log1p CPM (`log1p_cpm`) layer computed on demand; if present, double transforms are skipped.

3. **Gene feature extraction:** For each gene, the pipeline sums ATAC counts within ±10 kb windows, binned at 500 bp. Feature matrices are built on demand for each execution.
4. **Expression filtering:** Genes must have at least `min_cells_per_gene` cells above `min_expression` (defaults: 100 cells, 0.0 expression).
5. **k-NN smoothing:** Each cell is smoothed by averaging with its k nearest neighbors (k = `smoothing_k - 1`, default 19) using PCA-informed neighbor search within each split to reduce sparsity while maintaining dataset size.
6. **Pseudobulk (optional):** If `pseudobulk_group_size > 1`, PCA-guided, group-aware pooling within each `group_key` (default `sample`) produces meta-cells of the requested size.
7. **Scaling:** Feature scalers run on the training split; target scaling is skipped automatically when expression values are already log-transformed (set `force_target_scaling=True` to override).
8. **Splitting:** Train/val/test fractions default to 0.70/0.15/0.15 with `GroupShuffleSplit` by `group_key` (fallback to random splits when too few groups).
9. **Cross-validation:** Within the training split, models run 5-fold CV grouped by `group_key` when possible (else shuffled KFold) before fitting on the full training set.

### Inference on New ATAC Data

Use the inference helper to generate predictions from a trained run directory:

```bash
python -m spear.predict \
  --run-dir output/results/<run_name> \
  --model mlp \
  --atac-path /path/to/new_atac.h5ad \
  --output /path/to/predictions_inference.csv
```

### Troubleshooting

- Missing data paths: confirm `--atac-path`, `--rna-path`, and `--gtf-path` or the default `data/` layout.

## References

- `docs/config_reference.md`
- `docs/master_runbook.md`
- `docs/mouse_esc_dataset.md`
- `docs/endothelial_dataset.md`
- Barcode mismatch: ensure ATAC/RNA `obs_names` overlap and refer to the same cells.
- Memory pressure: reduce `--max-genes`, increase `--bin-size-bp`, or use chunking to shrink feature matrices.

## Documentation

- Dataset notes: `docs/mouse_esc_dataset.md`, `docs/endothelial_dataset.md`
- CLI/config reference: `docs/config_reference.md`
- Runbook (ops-focused): `docs/master_runbook.md`

## Citation

If you use SPEAR, cite the original data sources and this repository ([https://github.com/UzunLab/SPEAR](https://github.com/UzunLab/SPEAR)). A formal software citation can be added once a DOI is available.

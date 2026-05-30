# SPEAR: **S**ingle-cell-based **P**rediction of Gene **E**xpression from Chromatin **A**ccessibility **R**eadouts

![SPEAR Logo](docs/images/spear_logo.png)

## Overview

SPEAR (Single-cell-based Prediction of Gene Expression from Chromatin Accessibility Readouts) is an end-to-end framework for per-gene cis-regulatory prediction and interpretation from paired single-cell ATAC/RNA data. The primary analysis mode predicts gene-by-gene expression from local chromatin accessibility and exports feature-importance/SHAP artifacts for peak/bin interpretation. Multi-output modeling remains available for explicit legacy or supplementary comparisons.

## Inputs

- Paired ATAC/RNA `h5ad` files with overlapping barcodes.
- A reference GTF containing gene annotations.
- Gene manifests or explicit gene lists.
- Optional JSON configuration via `--config-json`.

## Outputs

- Run artifacts under `output/results/` (metrics, predictions, histories, model files).
- Logs under `output/logs` (pipeline logs: `<run_name>.log`; scheduler logs commonly appear as `spear_<jobid>_<task>.out/.err` for array jobs).
- Figures under `analysis/figs`.

## Usage

### SPEAR Framework Overview

![SPEAR framework figure](docs/images/SPEAR%20Framework%20Figure_transparent%20background.png)

### Key Features

### Project Goals

- Predict per-gene RNA expression from local ATAC accessibility and compare feature bases such as fixed genomic bins and ATAC peaks.
- Keep simple baselines central. Ridge/Elastic Net, MLP/ResNet, Transformer, and one tree/boosting model should answer most primary manuscript questions before the full model zoo is used.
- Provide a modular supplementary model zoo spanning convolutional, recurrent, transformer, graph-based, deep & cross networks, gradient boosting (XGBoost, CatBoost), multilayer perceptron, tree ensembles, and linear baselines (Ridge, Elastic Net, Lasso, OLS) so controlled comparisons are one CLI flag away.
- Produce test-set diagnostics (scatter plots, per-gene Pearson summaries, epoch histories) while persisting raw predictions for reproducibility and downstream analysis.

- **Modal-specific normalization:**
  - ATAC/RNA matrices are subset to shared barcodes and aligned to a common ordering; by default ATAC uses TF–IDF (`tfidf` layer), with alternatives (`counts_per_million`, `log1p_cpm`, or none) available via the `atac_layer` setting.
  - RNA counts are converted to counts-per-million and log-transformed (`log1p_cpm` layer) if needed; double log1p transforms are skipped when a normalized layer already exists.
  - Optional `StandardScaler`/`MinMaxScaler` may be applied on features and targets (target scaling is skipped for log-transformed targets unless `force_target_scaling=True`).
- **k-NN smoothing:** When enabled, training smooths `train`, `val`, and `test` independently after splitting by averaging each cell with its k nearest neighbors (20-cell neighborhoods by default, including the cell itself) using PCA-informed nearest-neighbor search to reduce sparsity while maintaining dataset size. Inference applies the same smoothing to the full inference batch before prediction.
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
scripts/                     # Public helper scripts for data prep, smoke tests, reporting, and plotting
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

### Tiny Smoke Test

Verify the installation without downloading real data:

```bash
python scripts/create_tiny_example_data.py
spear \
  --rna-path data/examples/tiny/tiny_RNA.h5ad \
  --atac-path data/examples/tiny/tiny_ATAC.h5ad \
  --gtf-path data/examples/tiny/tiny.gtf \
  --gene-manifest data/examples/tiny/tiny_genes.csv \
  --models ridge \
  --device cpu \
  --k-folds 2 \
  --disable-smoothing \
  --min-cells-per-gene 5 \
  --run-name tiny_smoke_ridge
```

Expected smoke-test outputs are written under `output/results/tiny_smoke_ridge/`, including
`summary_metrics.csv`, `summary_metrics_per_gene.csv`, `run_configuration.json`, and model-level
files under `models/ridge/`.

### Data Requirements

- Paired ATAC/RNA `h5ad` files with overlapping barcodes.
- A reference GTF containing gene annotations.
- See `docs/mouse_esc_dataset.md` and `docs/endothelial_dataset.md` for dataset-specific provenance and storage conventions.
- Use explicit gene manifests for controlled comparisons; `docs/run_inventory_schema.json` documents the run-inventory fields used to separate strict common-manifest rankings from fallback or feasibility runs.

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

SPEAR runs on local machines, cloud VMs, and HPC clusters. For schedulers such as Slurm, wrap the same `spear` command in your site-specific submission template and keep private accounts, partitions, and paths outside the public repository.

More flags and defaults are documented in `docs/config_reference.md`.

Environment / CLI highlights:

- `--gene-manifest` guarantees that every model trains on the same gene subset.
- `--cache-dir` enables on-disk reuse of preprocessing (recommended for repeated model runs).
- `--chromosomes genome-wide` explicitly disables chromosome filtering; provide a list to restrict loci.
- `--run-name` customises the output directory name (default: `spear_<model>_<genes>_<dataset>_<cpu/gpu>_<timestamp>`; missing components are omitted).
- W&B run name defaults to `<model>_<genes>_<dataset>` unless `--wandb-run-name` is provided.
- `--device` supports `cuda`, `cpu`, or `auto` (prefers CUDA when available; falls back otherwise).
- `--disable-pseudobulk` is a quick toggle to benchmark true single-cell training (equivalent to setting `--pseudobulk-group-size 1`).
- `--smoothing-target {all_splits,train_only,none}` and `--no-smoothing-y` let you evaluate whether smoothing should affect all splits, only training features, or features without RNA target averaging.
- `--global-atac-components <N>` appends global ATAC cell-state SVD components to each gene-local feature set; the default `0` keeps this disabled.
- `--fast-classical-mode` applies a faster profile for heavy classical multi-output models (`svr`, `lasso`, `elastic_net`, `hist_gradient_boosting`, `catboost`), useful when long CPU jobs are timing out.
- `--atac-layer` lets you swap CPM for alternative ATAC transforms such as `tfidf` or disable normalisation entirely.
- Torch CNN, ResNet, MLP, DCN, Transformer, and Graph models use RMSNorm-style normalization in their dense, attention, and convolutional blocks when available.
- ResNet now supports configurable squeeze-excitation attention: `--resnet-attention {se,none}` and `--resnet-attention-se-reduction <int>` (default: `se`, reduction `8`).

### Generate Gene Manifests

Generate the current per-sample 1000-gene manifests with:

```bash
python scripts/generate_all_sample_manifests.py \
  --base-dir data \
  --gene-count 1000 \
  --random-state 42
```

This writes three manifests per processed sample:

- `*_random_1000.csv`
- `*_hvg_1000.csv`
- `*_low_noisy_1000.csv`

### Example runs

```bash
# Per-gene baselines on CPU (default mode)
spear --models ridge lasso ols --device cpu --run-name per_gene_baselines

# Multi-output torch run with smaller smoothing and no pseudobulk
spear --multi-output --models mlp transformer --smoothing-k 5 --disable-pseudobulk --run-name multi_output_no_bulk

# ResNet with SE attention (default) and tuned reduction ratio
spear --models resnet --resnet-attention se --resnet-attention-se-reduction 4 --run-name resnet_se_r4

# ResNet ablation without attention
spear --models resnet --resnet-attention none --run-name resnet_no_attention
```

Per-gene training is now the default analysis path. Multi-output remains available for explicit comparisons and legacy runs via `--multi-output`.

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
- `shapley_values_mean.csv`, `shapley_values_mean.png`

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

SVR defaults to a linear kernel with configurable hyperparameters via `TrainingConfig.svr_*`.

### Utilities and scripts

- `scripts/create_tiny_example_data.py`: generate a tiny synthetic paired ATAC/RNA dataset for install smoke tests.
- `scripts/download_geo_raw_data.py`: download or stage GEO raw inputs and shared references under `data/`.
- `scripts/preprocess_geo_raw_to_spear.py`: canonical sample-first preprocessing entrypoint for embryonic and endothelial raw GEO inputs.
- `scripts/generate_all_sample_manifests.py`: build per-sample `random`, `hvg`, and `low_noisy` 1000-gene manifests for current processed outputs.
- `scripts/prepare_10x_pbmc_multiome.py`: prepare the public 10x PBMC multiome dataset into SPEAR-compatible RNA/ATAC files and manifests.
- `scripts/preflight_check.py`: quick readiness probe (env, packages, data paths, GTF). Run `python scripts/preflight_check.py --help` for options.
- `scripts/generate_all_reports.py`: summarize completed runs into CSV and markdown reports.
- `scripts/plot_prediction_structure.py`: compare real versus predicted RNA cell structure from exported raw predictions.

### Dependencies

All dependencies (runtime, dev, and notebooks) are listed in `requirements.txt`.

## Citation

If you use SPEAR in a publication, please cite the SPEAR manuscript/preprint and include the GitHub repository URL so others can reproduce the software version used in your analysis.

### Preprocessing Details

The current canonical preprocessing workflow is:

1. `python scripts/download_geo_raw_data.py --datasets ...`
2. `python scripts/preprocess_geo_raw_to_spear.py --datasets ...`
3. `python scripts/generate_all_sample_manifests.py --base-dir data --gene-count 1000`

1. **AnnData loading:** ATAC and RNA `h5ad` matrices are loaded through `anndata`, subset to shared barcodes, and aligned to a common ordering.
2. **Modal layers:**
   - ATAC: TF–IDF is created by default (`training.atac_layer='tfidf'`); alternative transforms (such as CPM) can be requested via configuration.
   - RNA: log1p CPM (`log1p_cpm`) layer computed on demand; if present, double transforms are skipped.

3. **Gene feature extraction:** For each gene, the pipeline sums ATAC counts within ±10 kb windows, binned at 500 bp. Feature matrices are built on demand for each execution.
4. **Expression filtering:** Genes must have at least `min_cells_per_gene` cells above `min_expression` (defaults: 100 cells, 0.0 expression).
5. **k-NN smoothing:** After train/val/test splitting, `train`, `val`, and `test` are each smoothed independently by averaging each cell with its k nearest neighbors (20-cell neighborhoods by default, including the cell itself) using PCA-informed neighbor search within that split to reduce sparsity while maintaining dataset size.
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

Inference reuses the saved training preprocessing settings. If smoothing was enabled for the run, the inference feature matrix is k-NN smoothed as a single batch before feature scaling and prediction.

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

If you use SPEAR in a publication, cite this repository and the original dataset/source publications listed in the dataset docs. A formal software citation can be added once a DOI is available.

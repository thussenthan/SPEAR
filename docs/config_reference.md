# SPEAR Configuration Reference

## Overview

This document outlines every user-facing knob in SPEAR (Single-cell-based Prediction of Gene Expression from Chromatin Accessibility Readouts), implemented in the `spear` package, including CLI flags, training defaults, and model hyperparameters.

## Inputs

- CLI flags via `spear` or `python -m spear.cli`.
- Optional JSON configuration via `--config-json`.

## Outputs

- Parsed configuration used by the training pipeline.
- No files are written directly by this reference.

## Usage

### Command Line Interface

The CLI entrypoint is `spear` (or `python -m spear.cli`). The table below lists all flags and
their defaults.

| Flag                              | Type      | Default                                                | Description                                                                                              |
| --------------------------------- | --------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `--base-dir`                      | path      | current working directory                              | Project root used to resolve data and output paths.                                                      |
| `--atac-path`                     | path      | `None`                                                 | Override ATAC AnnData path (h5ad).                                                                       |
| `--rna-path`                      | path      | `None`                                                 | Override RNA AnnData path (h5ad).                                                                        |
| `--gtf-path`                      | path      | `None`                                                 | Override GTF annotation path.                                                                            |
| `--genes`                         | list[str] | `None`                                                 | Explicit list of genes to process. Overrides random sampling.                                            |
| `--gene-manifest`                 | path      | `None`                                                 | Path to newline-/CSV-delimited gene manifest.                                                            |
| `--chromosomes`                   | list[str] | `None`                                                 | Filter genes by chromosome. Accepts `genome-wide`/`all` to disable filters.                              |
| `--max-genes`                     | int       | `None`                                                 | Hard cap on genes processed in per-gene mode. Multi-output uses full filtered set unless explicitly set. |
| `--models`                        | list[str] | `['cnn','rnn','lstm','mlp','xgboost','random_forest']` | Replace the default model roster (graph, catboost, etc. can be specified explicitly).                    |
| `--extra-models`                  | list[str] | `[]`                                                   | Extend the model roster without replacing defaults.                                                      |
| `--k-folds`                       | int       | `5`                                                    | Number of CV folds.                                                                                      |
| `--train-fraction`                | float     | `0.70`                                                 | Train split proportion (per-gene mode).                                                                  |
| `--val-fraction`                  | float     | `0.15`                                                 | Validation split proportion.                                                                             |
| `--test-fraction`                 | float     | `0.15`                                                 | Test split proportion.                                                                                   |
| `--group-key`                     | str       | `sample`                                               | AnnData obs column to use for grouped splits (use `none` to disable).                                    |
| `--window-bp`                     | int       | `10000`                                                | ATAC window size around each TSS.                                                                        |
| `--bin-size-bp`                   | int       | `500`                                                  | ATAC bin size (peak resolution).                                                                         |
| `--multioutput-feature-basis`     | enum      | `bin`                                                  | Feature basis for multi-output mode (`bin` for shared bins, `peak` for shared peaks).                    |
| `--scaler`                        | enum      | `standard`                                             | Feature scaler (`standard`, `minmax`, `none`).                                                           |
| `--target-scaler`                 | enum      | `standard`                                             | Target scaler (`standard`, `minmax`, `none`).                                                            |
| `--force-target-scaling`          | flag      | `False`                                                | Apply target scaling even when targets are already log-transformed.                                      |
| `--epochs`                        | int       | `100`                                                  | Training epochs for neural models.                                                                       |
| `--learning-rate`                 | float     | `1e-3`                                                 | Optimizer learning rate (torch models).                                                                  |
| `--lr-scheduler`                  | enum      | `cosine`                                               | LR scheduler for torch models (`none`, `cosine`).                                                         |
| `--warmup-epochs`                 | int       | `5`                                                    | Warmup epochs for LR scheduling.                                                                         |
| `--warmup-ratio`                  | float     | `0.1`                                                  | Warmup ratio fallback when warmup epochs are not set.                                                    |
| `--min-lr-ratio`                  | float     | `0.01`                                                 | Minimum LR floor as a ratio of base LR for cosine decay.                                                 |
| `--batch-size`                    | int       | `512`                                                  | Mini-batch size (torch models).                                                                          |
| `--gradient-accumulation-steps`   | int       | `1`                                                    | Number of micro-batches accumulated before each optimizer step.                                           |
| `--effective-batch-cap`           | int       | `48000`                                                | Cap used to scale effective batch size in multi-output mode (`cap // target_dim`).                       |
| `--pseudobulk-group-size`         | int       | `1`                                                    | Cells per pseudobulk neighborhood.                                                                       |
| `--pseudobulk-pca-components`     | int       | `10`                                                   | PCA components for pseudobulk grouping.                                                                  |
| `--disable-pseudobulk`            | flag      | `False`                                                | Shortcut to set `pseudobulk_group_size=1`.                                                               |
| `--smoothing-k`                   | int       | `20`                                                   | Neighborhood size for k-NN smoothing (use 1 to disable).                                                 |
| `--smoothing-pca-components`      | int       | `10`                                                   | PCA components for k-NN smoothing.                                                                       |
| `--disable-smoothing`             | flag      | `False`                                                | Disable k-NN smoothing of cells.                                                                         |
| `--fast-classical-mode`           | flag      | `False`                                                | Faster profile for heavy classical multi-output models (svr/lasso/elastic_net/hist_gradient_boosting/catboost). |
| `--resource-sample-seconds`       | float     | `60`                                                   | Interval (seconds) between resource usage samples.                                                       |
| `--transformer-embed-dim`         | int       | `128`                                                  | Transformer embedding dimension.                                                                         |
| `--transformer-num-layers`        | int       | `2`                                                    | Number of transformer encoder layers.                                                                    |
| `--transformer-dropout`           | float     | `0.2`                                                  | Transformer dropout for encoder and prediction head.                                                     |
| `--transformer-num-heads`         | int       | `None` (auto)                                          | Transformer attention heads (must divide `transformer_embed_dim`).                                       |
| `--resnet-attention`              | enum      | `se`                                                   | ResNet attention type (`se` for squeeze-excitation, `none` to disable).                                  |
| `--resnet-attention-se-reduction` | int       | `8`                                                    | Reduction ratio for ResNet squeeze-excitation bottleneck.                                                |
| `--enable-feature-importance`     | flag      | `False`                                                | Enable feature importance computation.                                                                   |
| `--feature-importance-samples`    | int       | `None` (all samples)                                   | Max samples for feature importance computation.                                                          |
| `--feature-importance-batch-size` | int       | `256`                                                  | Batch size for feature importance gradient accumulation.                                                 |
| `--enable-shap`                   | flag      | `False`                                                | Enable SHAP attribution export for torch models.                                                         |
| `--shap-max-samples`              | int       | `500`                                                  | Max samples to evaluate SHAP on.                                                                         |
| `--shap-background-samples`       | int       | `100`                                                  | Background samples used for SHAP baselines.                                                              |
| `--export-raw-predictions`        | flag      | `False`                                                | Export per-cell predictions (adds runtime and output size).                                              |
| `--device`                        | enum      | `cuda`                                                 | Run device (`cuda`, `cpu`, `auto`).                                                                      |
| `--atac-layer`                    | enum      | `tfidf`                                                | ATAC normalization (`counts_per_million`, `tfidf`, `log1p_cpm`, `none`).                                 |
| `--run-name`                      | str       | `spear_<model>_<genes>_<dataset>_<cpu/gpu>_<timestamp>` | Output run directory name.                                                                               |
| `--wandb`                         | flag      | `False`                                                | Enable Weights & Biases logging (requires `wandb` + login).                                              |
| `--wandb-project`                 | str       | `SPEAR`                                                | W&B project name.                                                                                        |
| `--wandb-entity`                  | str       | `None`                                                 | W&B entity/team name.                                                                                    |
| `--wandb-run-name`                | str       | `None`                                                 | Override W&B run name (defaults to `<model>_<genes>_<dataset>`).                                          |
| `--wandb-tags`                    | list[str] | `None`                                                 | Optional W&B tags.                                                                                       |
| `--wandb-group`                   | str       | `None`                                                 | Optional W&B group.                                                                                      |
| `--wandb-job-type`                | str       | `None`                                                 | Optional W&B job type.                                                                                   |
| `--wandb-no-artifacts`            | flag      | `False`                                                | Disable W&B artifact logging.                                                                            |
| `--wandb-no-tables`               | flag      | `False`                                                | Disable W&B table logging.                                                                               |
| `--wandb-no-media`                | flag      | `False`                                                | Disable W&B media logging (plots/images).                                                                |
| `--wandb-no-predictions-table`    | flag      | `False`                                                | Disable W&B predictions table logging.                                                                   |
| `--wandb-sweep`                   | flag      | `False`                                                | Apply sweep overrides from `wandb.config`.                                                               |
| `--wandb-table-max-rows`          | int       | `5000`                                                 | Max rows to log per W&B Table.                                                                           |
| `--wandb-media-max-items`         | int       | `50`                                                   | Max number of media items to log per run.                                                                |
| `--chunk-index`                   | int       | `0`                                                    | Gene chunk index (zero-based).                                                                           |
| `--chunk-total`                   | int       | `1`                                                    | Number of gene chunks.                                                                                   |
| `--config-json`                   | path      | `None`                                                 | Load pipeline configuration from JSON file.                                                              |
| `--per-gene`                      | flag      | `False`                                                | Run per-gene training (one model per gene).                                                              |
| `--multi-output`                  | flag      | `True` (unless `--per-gene`)                           | Enable cell-wise multi-output regression (default unless `--per-gene` is set).                           |
| `--rf-n-estimators`               | int       | `None` (model-specific default)                        | Override random forest tree count.                                                                       |
| `--rf-max-depth`                  | int       | `None`                                                 | Override maximum depth.                                                                                  |
| `--rf-min-samples-leaf`           | int       | `None`                                                 | Override leaf size.                                                                                      |
| `--rf-max-features`               | float/str | `None`                                                 | Override feature fraction.                                                                               |
| `--rf-bootstrap`                  | bool      | `None`                                                 | Force bootstrap sampling on/off.                                                                         |

### Training Configuration Defaults

Values below come from `TrainingConfig` and apply unless overridden via CLI or JSON.

| Parameter                       | Default                        | Notes                                                                                                                             |
| ------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `window_bp`                     | `10000`                        | +/- bp around TSS.                                                                                                                |
| `bin_size_bp`                   | `500`                          | ATAC bin resolution.                                                                                                              |
| `multioutput_feature_basis`     | `bin`                          | Multi-output feature basis (`bin` for shared bins, `peak` for shared peaks).                                                      |
| `k_folds`                       | `5`                            | Cross-validation folds.                                                                                                           |
| `train_fraction`                | `0.70`                         | Per-gene mode train split.                                                                                                        |
| `val_fraction`                  | `0.15`                         | Per-gene mode validation split.                                                                                                   |
| `test_fraction`                 | `0.15`                         | Per-gene mode test split.                                                                                                         |
| `batch_size`                    | `512`                          | Torch mini-batch size.                                                                                                            |
| `effective_batch_cap`           | `48000`                        | Cap used to scale effective batch size in multi-output mode (`cap // target_dim`).                                                |
| `epochs`                        | `100`                          | Torch training epochs.                                                                                                            |
| `learning_rate`                 | `1e-3`                         | Adam learning rate.                                                                                                               |
| `lr_scheduler`                  | `cosine`                       | Learning-rate scheduling strategy (`none` or `cosine`).                                                                           |
| `warmup_epochs`                 | `5`                            | Warmup in epochs before cosine decay.                                                                                             |
| `warmup_ratio`                  | `0.1`                          | Warmup fraction fallback used when warmup epochs are zero.                                                                        |
| `min_lr_ratio`                  | `0.01`                         | Minimum LR as ratio of base LR under cosine decay.                                                                                |
| `gradient_accumulation_steps`   | `1`                            | Number of gradient accumulation steps per optimizer update.                                                                        |
| `weight_decay`                  | `1e-5`                         | Adam weight decay.                                                                                                                |
| `max_grad_norm`                 | `5.0`                          | Gradient clipping for torch models (cnn/rnn/lstm/transformer/mlp/dcn/resnet/graph).                                                |
| `early_stopping_patience`       | `10`                           | Epoch patience on validation loss.                                                                                                |
| `random_state`                  | `42`                           | RNG seed for reproducibility.                                                                                                     |
| `device_preference`             | `cuda`                         | Preferred compute device.                                                                                                         |
| `scaler`                        | `standard`                     | Feature scaling (set `none` to disable).                                                                                          |
| `min_cells_per_gene`            | `100`                          | Minimum expressing cells per gene (per-gene mode).                                                                                |
| `min_expression`                | `0.0`                          | Raw expression threshold.                                                                                                         |
| `log1p_transform`               | `False`                        | Additional log1p on targets if raw layer selected.                                                                                |
| `target_scaler`                 | `standard`                     | Target scaling (skipped when `log1p_transform=True` or the RNA layer name starts with `log`, unless `force_target_scaling=True`). |
| `force_target_scaling`          | `False`                        | Apply target scaling even when targets are already log-transformed.                                                               |
| `enable_smoothing`              | `True`                         | Whether to apply k-NN smoothing within each split.                                                                                |
| `smoothing_k`                   | `20`                           | Neighborhood size for smoothing (use 1 to disable).                                                                               |
| `smoothing_pca_components`      | `10`                           | PCA components for smoothing neighbor search.                                                                                     |
| `pseudobulk_group_size`         | `1`                            | Cells per pseudobulk aggregate (1 disables pooling).                                                                              |
| `pseudobulk_pca_components`     | `10`                           | PCA dims for pseudobulk neighborhood search.                                                                                      |
| `fast_classical_mode`           | `False`                        | When enabled, applies faster defaults for heavy classical multi-output models (reduced CV folds and lighter model settings).     |
| `transformer_embed_dim`         | `128`                          | Transformer embedding size.                                                                                                       |
| `transformer_num_layers`        | `2`                            | Transformer encoder depth.                                                                                                        |
| `transformer_dropout`           | `0.2`                          | Transformer dropout for attention/MLP and final head.                                                                             |
| `transformer_num_heads`         | `None`                         | Attention heads; auto-selected when unset, must divide embedding dim when set.                                                    |
| `resnet_attention`              | `se`                           | Attention module in ResNet blocks (`se` or `none`).                                                                                |
| `resnet_attention_se_reduction` | `8`                            | Reduction ratio used by ResNet squeeze-excitation blocks.                                                                          |
| `min_expression_fraction`       | `0.10`                         | Fraction of cells expressing a gene for multi-output sampling.                                                                    |
| `enable_feature_importance`     | `False`                        | Whether to compute feature importance for torch models.                                                                           |
| `feature_importance_samples`    | `None` (all)                   | Max samples for feature importance computation.                                                                                   |
| `feature_importance_batch_size` | `256`                          | Batch size for feature importance gradient accumulation.                                                                          |
| `enable_shap`                   | `False`                        | Whether to compute SHAP attributions for torch models (multi-output returns one vector aggregated across outputs).                |
| `shap_max_samples`              | `500`                          | Max samples to evaluate SHAP on.                                                                                                  |
| `shap_background_samples`       | `100`                          | Background samples used for SHAP baselines.                                                                                       |
| `export_raw_predictions`        | `False`                        | Whether to export per-cell prediction CSVs.                                                                                       |
| `resource_sample_seconds`       | `60.0`                         | Interval between resource usage samples.                                                                                          |
| `catboost_iterations`           | `1000`                         | CatBoost iteration budget (when model is enabled).                                                                                |
| `rf_n_estimators`               | `None`                         | Falls back to model defaults below.                                                                                               |
| `rf_max_depth`                  | `None`                         | Unlimited depth when unset.                                                                                                       |
| `rf_min_samples_leaf`           | `None`                         | Model default when unset.                                                                                                         |
| `rf_max_features`               | `None`                         | Model default when unset.                                                                                                         |
| `rf_bootstrap`                  | `None`                         | Model default when unset.                                                                                                         |
| `svr_kernel`                    | `linear`                       | SVR kernel (`linear`, `rbf`, etc.).                                                                                               |
| `svr_C`                         | `1.0`                          | SVR regularization strength.                                                                                                      |
| `svr_epsilon`                   | `0.1`                          | Epsilon-insensitive loss parameter.                                                                                               |
| `svr_max_iter`                  | `50000`                        | Maximum SVR iterations.                                                                                                           |
| `svr_tol`                       | `1e-4`                         | SVR solver tolerance.                                                                                                             |
| `track_history`                 | `True`                         | Record training curves.                                                                                                           |
| `history_metrics`               | `['mse','pearson','spearman']` | Metrics tracked per epoch.                                                                                                        |
| `group_key`                     | `'sample'`                     | obs column used to group splits and CV folds.                                                                                     |
| `atac_layer`                    | `'tfidf'`                      | ATAC normalization layer (`counts_per_million`, `tfidf`, etc., or `None`).                                                        |
| `rna_expression_layer`          | `'log1p_cpm'`                  | RNA normalization layer (layers starting with `log` are treated as already log-transformed for scaling logic).                    |

### W&B Configuration (JSON)

If you load configuration via `--config-json`, the optional `wandb` block can enable W&B logging:

```json
{
  "wandb": {
    "enabled": true,
    "project": "SPEAR",
    "entity": "my-team",
    "run_name": "custom_run_name",
    "tags": ["dev", "sweep"],
    "group": "experiment_group",
    "job_type": "train",
    "log_artifacts": true,
    "log_tables": true,
    "log_media": true,
    "log_predictions_table": true,
    "sweep_overrides": false,
    "table_max_rows": 5000,
    "media_max_items": 50
  }
}
```

SPEAR skips W&B logging when `wandb` is not installed or when no API key is available (for example, missing
`WANDB_API_KEY` or a `~/.netrc` entry).

### Paths and Outputs

`PathsConfig.from_base(base_dir)` resolves the following locations relative to the base directory:

| Path Attribute | Default Location                                 |
| -------------- | ------------------------------------------------ |
| `atac_path`    | `data/embryonic/processed/combined_ATAC_qc.h5ad` |
| `rna_path`     | `data/embryonic/processed/combined_RNA_qc.h5ad`  |
| `gtf_path`     | `data/references/GCF_000001635.27_genomic.gtf`   |
| `output_dir`   | `output/results`                                 |
| `logs_dir`     | `output/logs`                                    |
| `figures_dir`  | `analysis/figs`                                  |

Override paths via CLI flags (`--atac-path`, `--rna-path`, `--gtf-path`) or by supplying a custom JSON config to the pipeline.
Data files are not published with the repository; treat the defaults above as local path conventions.

### Model Defaults

#### Torch Models

| Model         | Architecture Summary                                                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `cnn`         | 1D CNN with 3 conv blocks (32/64/128 channels), adaptive pooling, 512-unit dense head, dropout 0.2.                                             |
| `resnet`      | 1D ResNet with strided stem, three residual stages (32→64→128 channels), squeeze-excitation attention in residual blocks (configurable), adaptive pooling, 256-unit dense head, dropout 0.2. |
| `rnn`         | Conv down-sampling followed by RNN (`hidden_size=96`, `num_layers=1`), dense head with dropout 0.2.                                             |
| `lstm`        | Same conv front-end as RNN, `hidden_size=128` LSTM, dense head with dropout 0.2.                                                                |
| `transformer` | Conv projection to 128 channels, adaptive pooling, configurable transformer encoder (`embed_dim`, `num_layers`, auto/explicit `num_heads`), dense head with GELU and configurable dropout. |
| `graph`       | Implicit 1D graph message-passing network that chunkifies ATAC bins, applies learned edge weights, and aggregates through residual MLP layers.  |
| `dcn`         | Deep & Cross Network with 3 cross layers and a 256→256→128 deep tower; concatenated outputs are linearly projected to targets.                  |
| `mlp`         | Fully connected stack: 256→256→128 with LayerNorm + ReLU + dropout 0.2, output layer sized to target dimension.                                 |

Torch optimizers use `Adam(lr=1e-3, weight_decay=1e-5)` with automatic mixed precision when CUDA is available, plus optional cosine LR decay with warmup.

#### Scikit-learn / XGBoost Models

## References

- `src/spear/cli.py`
- `src/spear/config.py`

| Model                    | Default Hyperparameters                                                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `svr`                    | `kernel='linear'`, `C=1.0`, `epsilon=0.1`, `max_iter=50_000`, `tol=1e-4`; multi-output uses `MultiOutputRegressor`. Configurable via `TrainingConfig.svr_*`. |
| `xgboost`                | `n_estimators=800`, `max_depth=6`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `tree_method='hist'`.                    |
| `catboost`               | `iterations=1000`, `depth=6`, `learning_rate=0.05`, `loss_function='RMSE'`, auto CPU/GPU detection, verbose disabled.                                        |
| `random_forest`          | `n_estimators=600`, `min_samples_leaf=2`, `max_features=None`, `bootstrap=True`, `oob_score=True`.                                                           |
| `extra_trees`            | `n_estimators=800`, unlimited depth, `min_samples_leaf=1`, `bootstrap=False`.                                                                                |
| `hist_gradient_boosting` | `learning_rate=0.05`, `max_depth=6`, `max_iter=600`, `max_leaf_nodes=64`, `min_samples_leaf=20`, `l2_regularization=1e-3`.                                   |
| `ridge`                  | `alpha=1.0`, with `StandardScaler`.                                                                                                                          |
| `elastic_net`            | Single-target: `alpha=0.05`, `l1_ratio=0.5`; multi-target: `alpha=0.05`, `l1_ratio=0.3`, both with `StandardScaler`.                                         |
| `lasso`                  | Single-target: `alpha=0.05`; multi-target: `alpha=0.05` via `MultiTaskLasso`; both include `StandardScaler`.                                                 |
| `ols`                    | Ordinary least squares (`LinearRegression`) preceded by `StandardScaler`.                                                                                    |

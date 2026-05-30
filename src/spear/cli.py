import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from .config import (
    ModelConfig,
    PipelineConfig,
    PathsConfig,
    TrainingConfig,
    WandbConfig,
)
from .evaluation import run_pipeline
from .logging_utils import configure_logging, get_logger
from .wandb_utils import infer_dataset_name


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SPEAR: Single-cell-based Prediction of Gene Expression from Chromatin Accessibility Readouts"
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path.cwd()),
        help="Project root directory (defaults to the current working directory)",
    )
    parser.add_argument("--atac-path", help="Override ATAC AnnData path (h5ad)")
    parser.add_argument("--rna-path", help="Override RNA AnnData path (h5ad)")
    parser.add_argument("--gtf-path", help="Override GTF annotation path")
    parser.add_argument("--genes", nargs="*", help="Specific gene names to model")
    parser.add_argument(
        "--gene-manifest", help="Path to newline-delimited list of gene names to model"
    )
    parser.add_argument(
        "--chromosomes",
        nargs="*",
        help="Limit processing to genes on specific chromosomes",
    )
    parser.add_argument(
        "--max-genes", type=int, help="Maximum number of genes to process"
    )
    parser.add_argument(
        "--models", nargs="+", help="Models to evaluate (override defaults)"
    )
    parser.add_argument(
        "--extra-models", nargs="*", help="Additional models to include"
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        help="Number of folds for cross-validation; use 0 or 1 to disable CV and keep only the train/val/test split",
    )
    parser.add_argument(
        "--train-fraction", type=float, help="Training fraction (default 0.7)"
    )
    parser.add_argument(
        "--val-fraction", type=float, help="Validation fraction (default 0.15)"
    )
    parser.add_argument(
        "--test-fraction", type=float, help="Test fraction (default 0.15)"
    )
    parser.add_argument(
        "--group-key",
        help="AnnData obs column to use for grouped splits (set to 'none' to disable grouped splitting)",
    )
    parser.add_argument(
        "--window-bp",
        type=int,
        help=f"Window around TSS in base pairs (default {TrainingConfig().window_bp})",
    )
    parser.add_argument(
        "--bin-size-bp", type=int, help="Bin size in base pairs (default 500)"
    )
    parser.add_argument(
        "--multioutput-feature-basis",
        choices=["bin", "peak"],
        help="Feature basis for multi-output mode: bin (default) or peak",
    )
    parser.add_argument(
        "--per-gene-feature-basis",
        choices=["bin", "peak"],
        help=(
            "Feature basis for per-gene mode: bin (default, fixed-size bins) or "
            "peak (one feature per ATAC peak in the TSS window)"
        ),
    )
    parser.add_argument(
        "--per-gene-peak-min-peaks",
        type=int,
        help=(
            "Minimum number of peaks required in the per-gene peak window. "
            "When --per-gene-feature-basis peak is used, genes below this threshold "
            "are skipped instead of falling back to bin features."
        ),
    )
    parser.add_argument(
        "--per-gene-peak-distance-encoding",
        choices=["none", "signed_linear", "rbf"],
        help=(
            "Optional explicit distance-to-TSS encoding for per-gene peak features. "
            "When enabled, torch sequence models receive additional channel(s) derived from "
            "the signed peak offset from the TSS."
        ),
    )
    parser.add_argument(
        "--per-gene-peak-distance-rbf-bases",
        type=int,
        help="Number of RBF distance bases when using --per-gene-peak-distance-encoding rbf.",
    )
    parser.add_argument(
        "--per-gene-peak-distance-rbf-gamma",
        type=float,
        help="RBF gamma (bandwidth) when using --per-gene-peak-distance-encoding rbf.",
    )
    parser.add_argument(
        "--multioutput-local-only",
        action="store_true",
        help="Constrain multi-output torch models to each target's local feature window",
    )
    parser.add_argument(
        "--scaler", choices=["standard", "minmax", "none"], help="Feature scaler"
    )
    parser.add_argument(
        "--target-scaler", choices=["standard", "minmax", "none"], help="Target scaler"
    )
    parser.add_argument(
        "--force-target-scaling",
        action="store_true",
        help="Apply target scaler even when expression values are already log transformed",
    )
    parser.add_argument(
        "--allow-negative-predictions",
        action="store_true",
        help=(
            "Disable the default nonnegative prediction floor. Intended only for "
            "diagnostics because RNA expression should be >= 0."
        ),
    )
    parser.add_argument(
        "--enable-zero-aware-predictions",
        action="store_true",
        help=(
            "Fit a train-split expressed-vs-zero classifier per gene and use it to "
            "suppress or shrink predictions for likely zero-expression cells."
        ),
    )
    parser.add_argument(
        "--zero-aware-threshold",
        type=float,
        help="Probability threshold for --enable-zero-aware-predictions mask mode (default 0.5).",
    )
    parser.add_argument(
        "--zero-aware-mode",
        choices=["mask", "multiply"],
        help="Zero-aware post-processing mode: mask low-probability cells or multiply predictions by expression probability.",
    )
    parser.add_argument(
        "--enable-prediction-calibration",
        action="store_true",
        help="Fit per-gene y_true = intercept + slope * y_pred on validation cells and apply before metrics/export.",
    )
    parser.add_argument("--epochs", type=int, help="Training epochs for neural models")
    parser.add_argument(
        "--learning-rate", type=float, help="Learning rate for neural models"
    )
    parser.add_argument(
        "--weight-decay", type=float, help="Weight decay for neural models"
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        help="Optimizer for torch models (default adamw; adamw decouples weight decay from adaptive LR)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Early stopping patience (epochs without validation improvement)",
    )
    parser.add_argument(
        "--min-epochs-before-early-stopping",
        type=int,
        help="Minimum epochs to run before early stopping can trigger",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine"],
        help="Learning-rate schedule for torch models (default cosine with warmup)",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, help="Warmup epochs for LR scheduling"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        help="Fallback warmup ratio if warmup epochs not set",
    )
    parser.add_argument(
        "--min-lr-ratio", type=float, help="Minimum LR ratio for cosine decay floor"
    )
    parser.add_argument("--batch-size", type=int, help="Batch size for neural models")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Number of micro-batches to accumulate before each optimizer step",
    )
    parser.add_argument(
        "--effective-batch-cap",
        type=int,
        help=(
            "Cap for the effective batch size in multi-output mode (default 48000). "
            "When many targets are predicted at once, the per-step workload can exceed "
            "batch_size; lower this to avoid OOMs or slowdowns, raise it to use more throughput."
        ),
    )
    parser.add_argument(
        "--pseudobulk-group-size",
        type=int,
        help="Cells per pseudobulk group (pools/averages cells, reducing dataset size); use 1 to disable pooling",
    )
    parser.add_argument(
        "--pseudobulk-pca-components",
        type=int,
        help="Number of PCA components to build pseudobulk neighborhoods",
    )
    parser.add_argument(
        "--disable-pseudobulk",
        action="store_true",
        help="Disable pseudobulk pooling (equivalent to --pseudobulk-group-size 1)",
    )
    parser.add_argument(
        "--smoothing-k",
        type=int,
        help="Neighborhood size for k-NN smoothing (>=1). Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--smoothing-pca-components",
        type=int,
        help="PCA components for k-NN smoothing neighborhoods",
    )
    parser.add_argument(
        "--smoothing-target",
        choices=["train_only", "all_splits", "none"],
        help=(
            "Which splits receive k-NN feature smoothing: all_splits (default), "
            "train_only, or none. --disable-smoothing still disables smoothing globally."
        ),
    )
    parser.add_argument(
        "--no-smoothing-y",
        action="store_true",
        help="Smooth only feature matrices during k-NN smoothing; leave expression targets unchanged.",
    )
    parser.add_argument(
        "--global-atac-components",
        type=int,
        help=(
            "Append this many global ATAC cell-state SVD components to gene-local features "
            "(default 0 disables global features)."
        ),
    )
    parser.add_argument(
        "--disable-smoothing",
        action="store_true",
        help="Disable k-NN smoothing of cells",
    )
    parser.add_argument(
        "--fast-classical-mode",
        action="store_true",
        help=(
            "Apply a faster training profile for heavy classical multi-output models "
            "(svr/lasso/elastic_net/hist_gradient_boosting/catboost)."
        ),
    )
    parser.add_argument(
        "--resource-sample-seconds",
        type=float,
        help="Interval (in seconds) between resource usage samples (default 60)",
    )
    parser.add_argument(
        "--transformer-embed-dim", type=int, help="Transformer embedding dimension"
    )
    parser.add_argument(
        "--transformer-num-layers", type=int, help="Transformer encoder layer count"
    )
    parser.add_argument(
        "--transformer-dropout", type=float, help="Transformer dropout rate"
    )
    parser.add_argument(
        "--transformer-num-heads",
        type=int,
        help="Transformer attention heads (must divide transformer-embed-dim)",
    )
    parser.add_argument(
        "--transformer-arch",
        choices=["v1", "v2"],
        help="Transformer architecture variant (default v1)",
    )
    parser.add_argument(
        "--resnet-attention",
        choices=["none", "se"],
        help="ResNet attention type ('se' for squeeze-excitation, 'none' to disable)",
    )
    parser.add_argument(
        "--resnet-attention-se-reduction",
        type=int,
        help="Reduction ratio for ResNet SE attention MLP bottleneck",
    )
    parser.add_argument(
        "--torch-pearson-loss-weight",
        type=float,
        help="Optional Pearson-correlation loss weight added to torch-model training objective",
    )
    parser.add_argument(
        "--per-gene-cell-filter",
        choices=["auto", "on", "off"],
        help="Per-gene cell filtering mode: auto (default), on, or off",
    )
    parser.add_argument(
        "--min-cells-per-gene",
        type=int,
        help="Minimum cells required per gene after eligibility checks",
    )
    parser.add_argument(
        "--min-expression",
        type=float,
        help="Minimum expression threshold used for cell eligibility checks",
    )
    parser.add_argument(
        "--disable-per-gene-torch-stability-profile",
        action="store_true",
        help="Disable per-gene torch regularization overrides that reduce overfitting on small datasets",
    )
    parser.add_argument(
        "--per-gene-torch-epochs",
        type=int,
        help="Override epoch budget for per-gene torch models (default: use global --epochs)",
    )
    parser.add_argument(
        "--per-gene-torch-min-epochs",
        type=int,
        help="Override min-epochs-before-early-stopping for per-gene torch models",
    )
    parser.add_argument(
        "--per-model-batch-size",
        help=(
            "Per-model batch size overrides in 'model=size' format, comma-separated "
            "(e.g., 'transformer=8192,cnn=4096'). Falls back to --batch-size for unlisted models."
        ),
    )
    parser.add_argument(
        "--feature-prune-top-k",
        type=int,
        dest="feature_prune_top_k",
        help=(
            "After the first multi-output training pass, prune the feature matrix to the top-K "
            "most important features (by mean absolute importance across models) and retrain all "
            "models on the pruned dataset. Results saved under models_pruned/."
        ),
    )
    parser.add_argument(
        "--enable-feature-importance",
        action="store_true",
        help="Enable feature importance for multi-output torch models",
    )
    parser.add_argument(
        "--feature-importance-samples",
        type=int,
        help="Max samples for feature importance; omit for ALL samples (default: all)",
    )
    parser.add_argument(
        "--feature-importance-batch-size",
        type=int,
        help="Batch size for feature-importance gradient accumulation (default 128)",
    )
    parser.add_argument(
        "--enable-per-gene-panels",
        action="store_true",
        help="Generate per-gene feature-importance panel images",
    )
    parser.add_argument(
        "--enable-shap",
        action="store_true",
        help="Enable SHAP attribution export for multi-output torch models",
    )
    parser.add_argument(
        "--export-raw-predictions",
        action="store_true",
        help="Export per-cell predictions_raw.csv (adds runtime/output size)",
    )
    parser.add_argument(
        "--shap-max-samples",
        type=int,
        help="Max samples to evaluate SHAP on (default 500)",
    )
    parser.add_argument(
        "--shap-background-samples",
        type=int,
        help="Background samples used for SHAP baselines (default 100)",
    )
    parser.add_argument(
        "--atac-layer",
        choices=["counts_per_million", "tfidf", "log1p_cpm", "none"],
        help="ATAC normalization layer (default tfidf)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Preferred compute device (cuda, cpu, or auto to choose CUDA when available)",
    )
    parser.add_argument(
        "--dataset-name",
        help="Explicit dataset label for W&B grouping/filtering (auto-inferred when omitted)",
    )
    parser.add_argument("--run-name", help="Optional run name override")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires wandb + login)",
    )
    parser.add_argument("--wandb-project", help="W&B project name (default SPEAR)")
    parser.add_argument("--wandb-entity", help="W&B entity/team name")
    parser.add_argument(
        "--wandb-run-name", help="Override W&B run name (defaults to --run-name)"
    )
    parser.add_argument("--wandb-tags", nargs="*", help="Optional W&B tags")
    parser.add_argument("--wandb-group", help="Optional W&B group")
    parser.add_argument("--wandb-job-type", help="Optional W&B job type")
    parser.add_argument(
        "--wandb-no-artifacts",
        action="store_true",
        help="Disable W&B artifact logging when --wandb is enabled",
    )
    parser.add_argument(
        "--wandb-no-tables",
        action="store_true",
        help="Disable W&B table logging when --wandb is enabled",
    )
    parser.add_argument(
        "--wandb-no-media",
        action="store_true",
        help="Disable W&B media logging (plots/images) when --wandb is enabled",
    )
    parser.add_argument(
        "--wandb-no-predictions-table",
        action="store_true",
        help="Disable W&B predictions table logging when --wandb is enabled",
    )
    parser.add_argument(
        "--wandb-sweep",
        action="store_true",
        help="Apply sweep overrides from wandb.config when running as a sweep agent",
    )
    parser.add_argument(
        "--wandb-table-max-rows",
        type=int,
        help="Max rows to log per W&B Table (default 5000)",
    )
    parser.add_argument(
        "--wandb-media-max-items",
        type=int,
        help="Max number of media items to log per run (default 50)",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Zero-based index of the gene chunk to process",
    )
    parser.add_argument(
        "--chunk-total",
        type=int,
        default=1,
        help="Total number of gene chunks across all jobs",
    )
    parser.add_argument("--cache-dir", help="Directory for on-disk preprocessing cache")
    parser.add_argument(
        "--cache-scope",
        choices=["auto", "cellwise", "gene", "all", "none"],
        default="auto",
        help=(
            "Which preprocessing caches to use with --cache-dir. "
            "auto caches cell-wise/multi-output data only; use 'gene' or 'all' "
            "to opt into large per-gene disk caches."
        ),
    )
    parser.add_argument("--config-json", help="Path to configuration JSON file to load")
    parser.add_argument(
        "--per-gene",
        action="store_true",
        help="Run per-gene training (one model per gene); this is the default unless --multi-output is set",
    )
    parser.add_argument(
        "--multi-output",
        action="store_true",
        help="Explicitly enable cell-wise multi-output mode instead of the default per-gene mode",
    )
    parser.add_argument(
        "--rf-n-estimators", type=int, help="Number of trees for random forest models"
    )
    parser.add_argument(
        "--rf-max-depth", type=int, help="Maximum depth for random forest models"
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        help="Minimum samples per leaf for random forest models",
    )
    parser.add_argument(
        "--rf-max-features",
        help="Maximum features for random forest models (float fraction or keywords like sqrt)",
    )
    parser.add_argument(
        "--rf-bootstrap",
        choices=["true", "false"],
        help="Enable or disable bootstrap sampling for random forest models",
    )
    return parser


def _override_path(
    parser: argparse.ArgumentParser, current: Path, override: Optional[str], label: str
) -> Path:
    if not override:
        return current
    candidate = Path(override).expanduser().resolve()
    if not candidate.exists():
        parser.error(f"{label} not found at {candidate}")
    return candidate


def _maybe_use_endothelial_gtf(paths: PathsConfig, args: argparse.Namespace) -> None:
    if args.gtf_path:
        return
    hints = [args.atac_path or "", args.rna_path or "", args.gene_manifest or ""]
    if not any("endothelial" in hint.lower() for hint in hints):
        return
    human_gtf = (
        Path(args.base_dir).expanduser().resolve()
        / "data"
        / "references"
        / "gencode.v44.annotation.gtf.gz"
    )
    if human_gtf.exists():
        paths.gtf_path = human_gtf.resolve()


def _parse_rf_max_features(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _parse_per_model_batch_size(value: str) -> dict:
    result: dict = {}
    for pair in value.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        model, size_str = pair.split("=", 1)
        try:
            result[model.strip()] = int(size_str.strip())
        except ValueError:
            pass
    return result


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    training = TrainingConfig()

    direct_overrides = {
        "k_folds": args.k_folds,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "window_bp": args.window_bp,
        "bin_size_bp": args.bin_size_bp,
        "multioutput_feature_basis": args.multioutput_feature_basis,
        "per_gene_feature_basis": args.per_gene_feature_basis,
        "per_gene_peak_min_peaks": args.per_gene_peak_min_peaks,
        "per_gene_peak_distance_encoding": args.per_gene_peak_distance_encoding,
        "per_gene_peak_distance_rbf_bases": args.per_gene_peak_distance_rbf_bases,
        "per_gene_peak_distance_rbf_gamma": args.per_gene_peak_distance_rbf_gamma,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "warmup_epochs": args.warmup_epochs,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "early_stopping_patience": args.early_stopping_patience,
        "min_epochs_before_early_stopping": args.min_epochs_before_early_stopping,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_cap": args.effective_batch_cap,
        "smoothing_k": args.smoothing_k,
        "smoothing_pca_components": args.smoothing_pca_components,
        "smoothing_target": args.smoothing_target,
        "global_atac_components": args.global_atac_components,
        "pseudobulk_group_size": args.pseudobulk_group_size,
        "pseudobulk_pca_components": args.pseudobulk_pca_components,
        "resource_sample_seconds": args.resource_sample_seconds,
        "transformer_embed_dim": args.transformer_embed_dim,
        "transformer_num_layers": args.transformer_num_layers,
        "transformer_dropout": args.transformer_dropout,
        "transformer_num_heads": args.transformer_num_heads,
        "transformer_arch": args.transformer_arch,
        "resnet_attention": args.resnet_attention,
        "resnet_attention_se_reduction": args.resnet_attention_se_reduction,
        "torch_pearson_loss_weight": args.torch_pearson_loss_weight,
        "per_gene_cell_filter_mode": args.per_gene_cell_filter,
        "min_cells_per_gene": args.min_cells_per_gene,
        "min_expression": args.min_expression,
        "zero_aware_threshold": args.zero_aware_threshold,
        "zero_aware_mode": args.zero_aware_mode,
        "optimizer": args.optimizer,
        "per_gene_torch_epochs": args.per_gene_torch_epochs,
        "per_gene_torch_min_epochs": args.per_gene_torch_min_epochs,
        "feature_importance_samples": args.feature_importance_samples,
        "feature_importance_batch_size": args.feature_importance_batch_size,
        "shap_max_samples": args.shap_max_samples,
        "shap_background_samples": args.shap_background_samples,
        "rf_n_estimators": args.rf_n_estimators,
        "rf_max_depth": args.rf_max_depth,
        "rf_min_samples_leaf": args.rf_min_samples_leaf,
        "feature_prune_top_k": args.feature_prune_top_k,
    }
    for field_name, value in direct_overrides.items():
        if value is not None:
            setattr(training, field_name, value)

    if args.scaler:
        training.scaler = None if args.scaler == "none" else args.scaler
    if args.target_scaler:
        training.target_scaler = (
            None if args.target_scaler == "none" else args.target_scaler
        )
    if args.group_key is not None:
        training.group_key = (
            None if args.group_key.lower() == "none" else args.group_key
        )
    if args.atac_layer:
        training.atac_layer = None if args.atac_layer == "none" else args.atac_layer
    if args.rf_max_features is not None:
        training.rf_max_features = _parse_rf_max_features(args.rf_max_features)
    if args.rf_bootstrap is not None:
        training.rf_bootstrap = args.rf_bootstrap.lower() == "true"

    training.force_target_scaling = bool(args.force_target_scaling)
    if args.allow_negative_predictions:
        training.prediction_min_value = None
    training.enable_zero_aware_predictions = bool(args.enable_zero_aware_predictions)
    training.enable_prediction_calibration = bool(args.enable_prediction_calibration)
    training.enable_smoothing = not args.disable_smoothing
    training.smoothing_y = not bool(args.no_smoothing_y)
    training.fast_classical_mode = bool(args.fast_classical_mode)
    training.multioutput_local_only = bool(args.multioutput_local_only)
    training.per_gene_torch_stability_profile = not bool(
        args.disable_per_gene_torch_stability_profile
    )
    if args.per_model_batch_size:
        training.per_model_batch_size = _parse_per_model_batch_size(
            args.per_model_batch_size
        )
    training.device_preference = args.device
    training.enable_feature_importance = bool(args.enable_feature_importance)
    training.enable_per_gene_panels = bool(args.enable_per_gene_panels)
    training.enable_shap = bool(args.enable_shap)
    training.export_raw_predictions = bool(args.export_raw_predictions)
    if args.disable_pseudobulk:
        training.pseudobulk_group_size = 1

    training.validate()
    return training


def _build_wandb_config(args: argparse.Namespace) -> WandbConfig:
    wandb_config = WandbConfig()

    direct_overrides = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "run_name": args.wandb_run_name,
        "group": args.wandb_group,
        "job_type": args.wandb_job_type,
        "table_max_rows": args.wandb_table_max_rows,
        "media_max_items": args.wandb_media_max_items,
    }
    for field_name, value in direct_overrides.items():
        if value is not None:
            setattr(wandb_config, field_name, value)

    if args.wandb_tags:
        wandb_config.tags = list(args.wandb_tags)

    wandb_config.enabled = bool(args.wandb)
    wandb_config.log_artifacts = not args.wandb_no_artifacts
    wandb_config.log_tables = not args.wandb_no_tables
    wandb_config.log_media = not args.wandb_no_media
    wandb_config.log_predictions_table = not args.wandb_no_predictions_table
    wandb_config.sweep_overrides = bool(args.wandb_sweep)
    return wandb_config


def _resolve_multi_output_mode(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> bool:
    if args.multi_output and args.per_gene:
        parser.error("Cannot set both --multi-output and --per-gene")
    return bool(args.multi_output)


def _load_requested_genes(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[Optional[list[str]], Optional[Path]]:
    genes: Optional[list[str]] = list(args.genes) if args.genes else None
    manifest_path: Optional[Path] = None
    if not args.gene_manifest:
        return genes, manifest_path

    manifest_path = Path(args.gene_manifest).expanduser().resolve()
    if not manifest_path.exists():
        parser.error(f"Gene manifest not found at {manifest_path}")
    manifest_genes = _load_manifest_genes(manifest_path)
    if not manifest_genes:
        parser.error(f"Gene manifest {manifest_path} did not contain any gene entries")
    return manifest_genes, manifest_path


def _derive_run_name(config: PipelineConfig, explicit_run_name: Optional[str]) -> str:
    if explicit_run_name:
        return explicit_run_name

    dataset = infer_dataset_name(config)
    models_list = config.all_models()
    model = models_list[0] if models_list else None
    device = (config.training.device_preference or "").lower()
    if device not in {"cpu", "cuda", "auto"}:
        device = ""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gene_count = config.max_genes
    if gene_count is None:
        gene_count = len(config.genes) if config.genes else None
    gene_label = None if gene_count in (None, 0) else f"{gene_count}genes"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "_".join(
        part
        for part in ["spear", model, gene_label, dataset, device, timestamp]
        if part
    )


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv and argv[0] == "generate-manifest":
        from .manifest import generate_manifest_main

        raise SystemExit(generate_manifest_main(argv[1:]))

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.chunk_total < 1:
        parser.error("--chunk-total must be >= 1")
    if args.chunk_index < 0 or args.chunk_index >= args.chunk_total:
        parser.error("--chunk-index must satisfy 0 <= chunk_index < chunk_total")

    if args.config_json:
        config_path = Path(args.config_json).expanduser().resolve()
        payload = json.loads(config_path.read_text())
        try:
            config = _config_from_json(payload)
        except ValueError as exc:
            parser.error(f"Invalid configuration in {config_path}: {exc}")
    else:
        paths = PathsConfig.from_base(
            args.base_dir,
            atac_filename=args.atac_path or "combined_ATAC_qc.h5ad",
            rna_filename=args.rna_path or "combined_RNA_qc.h5ad",
            gtf_filename=args.gtf_path or "GCF_000001635.27_genomic.gtf",
        )

        paths.atac_path = _override_path(
            parser, paths.atac_path, args.atac_path, "ATAC path"
        )
        paths.rna_path = _override_path(
            parser, paths.rna_path, args.rna_path, "RNA path"
        )
        paths.gtf_path = _override_path(
            parser, paths.gtf_path, args.gtf_path, "GTF path"
        )
        _maybe_use_endothelial_gtf(paths, args)

        try:
            training = _build_training_config(args)
        except ValueError as exc:
            parser.error(f"Invalid training configuration: {exc}")

        models = ModelConfig()
        if args.models:
            models.model_names = args.models
        if args.extra_models:
            models.extra_models = args.extra_models

        multi_output_mode = _resolve_multi_output_mode(args, parser)
        gene_list, manifest_path = _load_requested_genes(args, parser)
        wandb_config = _build_wandb_config(args)

        config = PipelineConfig(
            paths=paths,
            training=training,
            models=models,
            wandb=wandb_config,
            genes=gene_list,
            chromosomes=list(args.chromosomes) if args.chromosomes else None,
            max_genes=args.max_genes,
            chunk_total=args.chunk_total,
            chunk_index=args.chunk_index,
            multi_output=multi_output_mode,
            gene_manifest_path=manifest_path if args.gene_manifest else None,
            dataset=args.dataset_name or None,
        )
        if args.cache_dir:
            config.cache_dir = Path(args.cache_dir)
            config.cache_scope = args.cache_scope

    run_name = _derive_run_name(config, args.run_name)
    config.run_name = run_name
    config.ensure_directories()

    log_path, run_context = configure_logging(config.paths.logs_dir, run_name)
    config.run_context = run_context
    config.log_path = log_path
    logger = get_logger(__name__)
    logger.info("Logging to %s", log_path)

    try:
        output_dir = run_pipeline(config)
    except Exception:
        logger.exception("Pipeline terminated with an error")
        logger.error("RUN_COMPLETE_STATUS=FAILURE")
        raise SystemExit(1)

    logger.info("Pipeline complete. Results stored in %s", output_dir)
    logger.info("RUN_COMPLETE_STATUS=SUCCESS")


def _config_from_json(payload: dict) -> PipelineConfig:
    base_dir = payload.get("base_dir", str(Path.cwd()))
    paths = PathsConfig.from_base(base_dir)
    training_payload = payload.get("training", {})
    models_payload = payload.get("models", {})
    wandb_payload = payload.get("wandb", {})

    training = TrainingConfig(**training_payload)
    training.validate()
    models = ModelConfig(**models_payload)
    wandb_config = WandbConfig(**wandb_payload)

    return PipelineConfig(
        paths=paths,
        training=training,
        models=models,
        wandb=wandb_config,
        genes=payload.get("genes"),
        chromosomes=payload.get("chromosomes"),
        max_genes=payload.get("max_genes"),
        chunk_total=payload.get("chunk_total", 1),
        chunk_index=payload.get("chunk_index", 0),
        # Default to per-gene unless explicitly enabled in JSON payload
        multi_output=payload.get("multi_output", False),
        cache_dir=Path(payload["cache_dir"]) if payload.get("cache_dir") else None,
        cache_scope=payload.get("cache_scope", "auto"),
        gene_manifest_path=(
            Path(payload["gene_manifest_path"])
            if payload.get("gene_manifest_path")
            else None
        ),
    )


def _load_manifest_genes(manifest_path: Path) -> list[str]:
    def _normalize_header(value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        return normalized

    text = manifest_path.read_text().splitlines()
    stripped = [
        line.strip()
        for line in text
        if line.strip() and not line.strip().startswith("#")
    ]
    if not stripped:
        return []

    sniff_sample = "\n".join(stripped[:5])
    if any(delim in sniff_sample for delim in (",", "\t", ";")):
        try:
            dialect = csv.Sniffer().sniff(sniff_sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.get_dialect("excel")
        genes: list[str] = []
        with manifest_path.open("r", newline="") as handle:
            reader = csv.reader(handle, dialect)
            rows = [row for row in reader if row]
        if not rows:
            return []

        header_candidates = {
            _normalize_header(value): idx for idx, value in enumerate(rows[0])
        }
        gene_col = None
        for key in ("gene_name", "gene", "gene_id", "geneid"):
            if key in header_candidates:
                gene_col = header_candidates[key]
                break
        start_idx = 1 if gene_col is not None else 0
        if gene_col is None:
            gene_col = 0
        for row in rows[start_idx:]:
            if gene_col < len(row):
                value = row[gene_col].strip()
                if value:
                    genes.append(value)
        unique_ordered = list(dict.fromkeys(genes))
        return unique_ordered

    # If the manifest is a single-column file without delimiters, drop a header line if present.
    if stripped:
        header = _normalize_header(stripped[0])
        if header in {"gene_name", "gene", "gene_id", "geneid", "symbol"}:
            stripped = stripped[1:]
    return list(dict.fromkeys(stripped))


if __name__ == "__main__":
    main()

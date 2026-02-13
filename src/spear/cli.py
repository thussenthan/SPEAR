
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from .config import ModelConfig, PipelineConfig, PathsConfig, TrainingConfig, WandbConfig
from .evaluation import run_pipeline
from .logging_utils import configure_logging, get_logger
from .wandb_utils import infer_dataset_name


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPEAR: Single-cell-based Prediction of Gene Expression from Chromatin Accessibility Readouts")
    parser.add_argument(
        "--base-dir",
        default=str(Path.cwd()),
        help="Project root directory (defaults to the current working directory)",
    )
    parser.add_argument("--atac-path", help="Override ATAC AnnData path (h5ad)")
    parser.add_argument("--rna-path", help="Override RNA AnnData path (h5ad)")
    parser.add_argument("--gtf-path", help="Override GTF annotation path")
    parser.add_argument("--genes", nargs="*", help="Specific gene names to model")
    parser.add_argument("--gene-manifest", help="Path to newline-delimited list of gene names to model")
    parser.add_argument("--chromosomes", nargs="*", help="Limit processing to genes on specific chromosomes")
    parser.add_argument("--max-genes", type=int, help="Maximum number of genes to process")
    parser.add_argument("--models", nargs="*", help="Models to evaluate (override defaults)")
    parser.add_argument("--extra-models", nargs="*", help="Additional models to include")
    parser.add_argument("--k-folds", type=int, help="Number of folds for cross-validation")
    parser.add_argument("--train-fraction", type=float, help="Training fraction (default 0.7)")
    parser.add_argument("--val-fraction", type=float, help="Validation fraction (default 0.15)")
    parser.add_argument("--test-fraction", type=float, help="Test fraction (default 0.15)")
    parser.add_argument(
        "--group-key",
        help="AnnData obs column to use for grouped splits (set to 'none' to disable grouped splitting)",
    )
    parser.add_argument("--window-bp", type=int, help="Window around TSS in base pairs (default 10,000)")
    parser.add_argument("--bin-size-bp", type=int, help="Bin size in base pairs (default 500)")
    parser.add_argument(
        "--multioutput-feature-basis",
        choices=["bin", "peak"],
        help="Feature basis for multi-output mode: bin (default) or peak",
    )
    parser.add_argument("--scaler", choices=["standard", "minmax", "none"], help="Feature scaler")
    parser.add_argument("--target-scaler", choices=["standard", "minmax", "none"], help="Target scaler")
    parser.add_argument(
        "--force-target-scaling",
        action="store_true",
        help="Apply target scaler even when expression values are already log transformed",
    )
    parser.add_argument("--epochs", type=int, help="Training epochs for neural models")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for neural models")
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine"],
        help="Learning-rate schedule for torch models (default cosine with warmup)",
    )
    parser.add_argument("--warmup-epochs", type=int, help="Warmup epochs for LR scheduling")
    parser.add_argument("--warmup-ratio", type=float, help="Fallback warmup ratio if warmup epochs not set")
    parser.add_argument("--min-lr-ratio", type=float, help="Minimum LR ratio for cosine decay floor")
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
        "--disable-smoothing",
        action="store_true",
        help="Disable k-NN smoothing of cells",
    )
    parser.add_argument(
        "--resource-sample-seconds",
        type=float,
        help="Interval (in seconds) between resource usage samples (default 60)",
    )
    parser.add_argument("--transformer-embed-dim", type=int, help="Transformer embedding dimension")
    parser.add_argument("--transformer-num-layers", type=int, help="Transformer encoder layer count")
    parser.add_argument("--transformer-dropout", type=float, help="Transformer dropout rate")
    parser.add_argument(
        "--transformer-num-heads",
        type=int,
        help="Transformer attention heads (must divide transformer-embed-dim)",
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
    parser.add_argument("--run-name", help="Optional run name override")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires wandb + login)",
    )
    parser.add_argument("--wandb-project", help="W&B project name (default SPEAR)")
    parser.add_argument("--wandb-entity", help="W&B entity/team name")
    parser.add_argument("--wandb-run-name", help="Override W&B run name (defaults to --run-name)")
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
    parser.add_argument("--chunk-index", type=int, default=0, help="Zero-based index of the gene chunk to process")
    parser.add_argument("--chunk-total", type=int, default=1, help="Total number of gene chunks across all jobs")
    parser.add_argument("--cache-dir", help="Directory for on-disk preprocessing cache")
    parser.add_argument("--config-json", help="Path to configuration JSON file to load")
    parser.add_argument(
        "--per-gene",
        action="store_true",
        help="Run per-gene training (one model per gene) instead of the default cell-wise multi-output mode",
    )
    parser.add_argument(
        "--multi-output",
        action="store_true",
        help="Explicitly enable cell-wise multi-output mode (default unless --per-gene is set)",
    )
    parser.add_argument("--rf-n-estimators", type=int, help="Number of trees for random forest models")
    parser.add_argument("--rf-max-depth", type=int, help="Maximum depth for random forest models")
    parser.add_argument("--rf-min-samples-leaf", type=int, help="Minimum samples per leaf for random forest models")
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


def main(argv: Optional[list[str]] = None) -> None:
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
        paths = PathsConfig.from_base(args.base_dir)

        def _override_path(current: Path, override: Optional[str], label: str) -> Path:
            if not override:
                return current
            candidate = Path(override).expanduser().resolve()
            if not candidate.exists():
                parser.error(f"{label} not found at {candidate}")
            return candidate

        paths.atac_path = _override_path(paths.atac_path, args.atac_path, "ATAC path")
        paths.rna_path = _override_path(paths.rna_path, args.rna_path, "RNA path")
        paths.gtf_path = _override_path(paths.gtf_path, args.gtf_path, "GTF path")
        # Dataset-aware GTF fallback: endothelial runs generally use human symbols.
        # If the user did not explicitly pass --gtf-path, prefer the bundled
        # human GTF when endothelial paths/manifests are detected.
        if not args.gtf_path:
            endothelial_hints = [
                args.atac_path or "",
                args.rna_path or "",
                args.gene_manifest or "",
            ]
            if any("endothelial" in hint.lower() for hint in endothelial_hints):
                base_root = Path(args.base_dir).expanduser().resolve()
                human_gtf = base_root / "data" / "references" / "gencode.v44.annotation.gtf.gz"
                if human_gtf.exists():
                    paths.gtf_path = human_gtf.resolve()

        training = TrainingConfig()
        if args.k_folds:
            training.k_folds = args.k_folds
        if args.train_fraction:
            training.train_fraction = args.train_fraction
        if args.val_fraction:
            training.val_fraction = args.val_fraction
        if args.test_fraction:
            training.test_fraction = args.test_fraction
        if args.window_bp:
            training.window_bp = args.window_bp
        if args.bin_size_bp:
            training.bin_size_bp = args.bin_size_bp
        if args.multioutput_feature_basis:
            training.multioutput_feature_basis = args.multioutput_feature_basis
        if args.scaler:
            training.scaler = None if args.scaler == "none" else args.scaler
        if args.target_scaler:
            training.target_scaler = None if args.target_scaler == "none" else args.target_scaler
        if args.force_target_scaling:
            training.force_target_scaling = True
        if args.group_key is not None:
            training.group_key = None if args.group_key.lower() == "none" else args.group_key
        if args.epochs:
            training.epochs = args.epochs
        if args.learning_rate:
            training.learning_rate = args.learning_rate
        if args.lr_scheduler:
            training.lr_scheduler = args.lr_scheduler
        if args.warmup_epochs is not None:
            training.warmup_epochs = args.warmup_epochs
        if args.warmup_ratio is not None:
            training.warmup_ratio = args.warmup_ratio
        if args.min_lr_ratio is not None:
            training.min_lr_ratio = args.min_lr_ratio
        if args.batch_size:
            training.batch_size = args.batch_size
        if args.gradient_accumulation_steps is not None:
            training.gradient_accumulation_steps = args.gradient_accumulation_steps
        if args.effective_batch_cap is not None:
            training.effective_batch_cap = args.effective_batch_cap
        if args.smoothing_k is not None:
            training.smoothing_k = args.smoothing_k
        if args.smoothing_pca_components is not None:
            training.smoothing_pca_components = args.smoothing_pca_components
        if args.disable_smoothing:
            training.enable_smoothing = False
        if args.pseudobulk_group_size is not None:
            training.pseudobulk_group_size = args.pseudobulk_group_size
        if args.disable_pseudobulk:
            training.pseudobulk_group_size = 1
        if args.pseudobulk_pca_components is not None:
            training.pseudobulk_pca_components = args.pseudobulk_pca_components
        if args.resource_sample_seconds is not None:
            training.resource_sample_seconds = args.resource_sample_seconds
        if args.atac_layer:
            training.atac_layer = None if args.atac_layer == "none" else args.atac_layer
        training.device_preference = args.device
        if args.enable_feature_importance:
            training.enable_feature_importance = True
        if args.feature_importance_samples is not None:
            training.feature_importance_samples = args.feature_importance_samples
        if args.feature_importance_batch_size is not None:
            training.feature_importance_batch_size = args.feature_importance_batch_size
        if args.enable_shap:
            training.enable_shap = True
        training.export_raw_predictions = args.export_raw_predictions
        if args.shap_max_samples is not None:
            training.shap_max_samples = args.shap_max_samples
        if args.shap_background_samples is not None:
            training.shap_background_samples = args.shap_background_samples
        if args.transformer_embed_dim is not None:
            training.transformer_embed_dim = args.transformer_embed_dim
        if args.transformer_num_layers is not None:
            training.transformer_num_layers = args.transformer_num_layers
        if args.transformer_dropout is not None:
            training.transformer_dropout = args.transformer_dropout
        if args.transformer_num_heads is not None:
            training.transformer_num_heads = args.transformer_num_heads
        if args.rf_n_estimators is not None:
            training.rf_n_estimators = args.rf_n_estimators
        if args.rf_max_depth is not None:
            training.rf_max_depth = args.rf_max_depth
        if args.rf_min_samples_leaf is not None:
            training.rf_min_samples_leaf = args.rf_min_samples_leaf
        if args.rf_max_features is not None:
            try:
                training.rf_max_features = float(args.rf_max_features)
            except ValueError:
                training.rf_max_features = args.rf_max_features
        if args.rf_bootstrap is not None:
            training.rf_bootstrap = args.rf_bootstrap.lower() == "true"
        try:
            training.validate()
        except ValueError as exc:
            parser.error(f"Invalid training configuration: {exc}")

        models = ModelConfig()
        if args.models:
            models.model_names = args.models
        if args.extra_models:
            models.extra_models = args.extra_models

        if args.multi_output and args.per_gene:
            parser.error("Cannot set both --multi-output and --per-gene")

        multi_output_mode = True
        if args.per_gene:
            multi_output_mode = False
        elif args.multi_output:
            multi_output_mode = True

        gene_list: Optional[list[str]] = list(args.genes) if args.genes else None
        if args.gene_manifest:
            manifest_path = Path(args.gene_manifest).expanduser().resolve()
            if not manifest_path.exists():
                parser.error(f"Gene manifest not found at {manifest_path}")
            manifest_genes = _load_manifest_genes(manifest_path)
            if not manifest_genes:
                parser.error(f"Gene manifest {manifest_path} did not contain any gene entries")
            gene_list = manifest_genes

        wandb_config = WandbConfig()
        if args.wandb:
            wandb_config.enabled = True
        if args.wandb_project:
            wandb_config.project = args.wandb_project
        if args.wandb_entity:
            wandb_config.entity = args.wandb_entity
        if args.wandb_run_name:
            wandb_config.run_name = args.wandb_run_name
        if args.wandb_tags:
            wandb_config.tags = list(args.wandb_tags)
        if args.wandb_group:
            wandb_config.group = args.wandb_group
        if args.wandb_job_type:
            wandb_config.job_type = args.wandb_job_type
        if args.wandb_no_artifacts:
            wandb_config.log_artifacts = False
        if args.wandb_no_tables:
            wandb_config.log_tables = False
        if args.wandb_no_media:
            wandb_config.log_media = False
        if args.wandb_no_predictions_table:
            wandb_config.log_predictions_table = False
        if args.wandb_sweep:
            wandb_config.sweep_overrides = True
        if args.wandb_table_max_rows is not None:
            wandb_config.table_max_rows = args.wandb_table_max_rows
        if args.wandb_media_max_items is not None:
            wandb_config.media_max_items = args.wandb_media_max_items

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
        )
        if args.cache_dir:
            config.cache_dir = Path(args.cache_dir)

    if args.run_name:
        run_name = args.run_name
    else:
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
        parts = ["spear", model, gene_label, dataset, device, timestamp]
        run_name = "_".join([part for part in parts if part])
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
        # Default to multi-output unless explicitly disabled in JSON payload
        multi_output=payload.get("multi_output", True),
        cache_dir=Path(payload["cache_dir"]) if payload.get("cache_dir") else None,
    )


def _load_manifest_genes(manifest_path: Path) -> list[str]:
    text = manifest_path.read_text().splitlines()
    stripped = [line.strip() for line in text if line.strip() and not line.strip().startswith("#")]
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
        def _normalize_header(value: str) -> str:
            normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
            return normalized

        header_candidates = {_normalize_header(value): idx for idx, value in enumerate(rows[0])}
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

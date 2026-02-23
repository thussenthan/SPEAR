
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PathsConfig:
    base_dir: Path
    atac_path: Path
    rna_path: Path
    gtf_path: Path
    output_dir: Path
    logs_dir: Path
    figures_dir: Path

    @classmethod
    def from_base(
        cls,
        base_dir: str | Path,
        atac_filename: str = "combined_ATAC_qc.h5ad",
        rna_filename: str = "combined_RNA_qc.h5ad",
        gtf_filename: str = "GCF_000001635.27_genomic.gtf",
    ) -> "PathsConfig":
        root = Path(base_dir).expanduser().resolve()

        def _resolve(filename: str, fallback_dirs: list[str]) -> Path:
            candidate = (root / filename).expanduser()
            if candidate.exists():
                return candidate.resolve()
            for rel_dir in fallback_dirs:
                alt = (root / rel_dir / filename).expanduser()
                if alt.exists():
                    return alt.resolve()
            searched = [str((root / filename).expanduser())] + [
                str((root / rel_dir / filename).expanduser()) for rel_dir in fallback_dirs
            ]
            raise FileNotFoundError(
                f"Could not locate '{filename}'. Looked in: {', '.join(searched)}"
            )

        fallback_data_dirs = ["data/embryonic/processed", "data/raw"]
        atac_path = _resolve(atac_filename, fallback_data_dirs)
        rna_path = _resolve(rna_filename, fallback_data_dirs)
        gtf_path = _resolve(gtf_filename, ["data/references"])
        output_root = (root / "output").resolve()
        output_dir = (output_root / "results").resolve()
        logs_dir = (output_root / "logs").resolve()
        figures_dir = (root / "analysis" / "figs").resolve()
        return cls(root, atac_path, rna_path, gtf_path, output_dir, logs_dir, figures_dir)


@dataclass
class TrainingConfig:
    window_bp: int = 10_000
    bin_size_bp: int = 500
    k_folds: int = 5
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 512
    # Cap for the effective batch size in multi-output mode.
    # When predicting many targets at once, the training loop may expand the
    # per-step workload beyond batch_size (e.g., batch_size × outputs). This
    # cap limits that effective size to control memory/compute; lower it if you
    # see OOMs or slowdowns, raise it to use more throughput when resources allow.
    # Rule of thumb: start in the 10_000–50_000 range (e.g., ~batch_size × 50–100)
    # on a 16–24 GB GPU; halve this on smaller GPUs or if you hit OOMs, and
    # increase gradually if utilization is low and memory headroom is available.
    effective_batch_cap: int = 48_000
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01
    gradient_accumulation_steps: int = 1
    transformer_embed_dim: int = 128
    transformer_num_layers: int = 2
    transformer_dropout: float = 0.2
    transformer_num_heads: Optional[int] = None
    transformer_arch: str = "v1"
    resnet_attention: str = "se"
    resnet_attention_se_reduction: int = 8
    torch_pearson_loss_weight: float = 0.0
    # Gradient clipping for torch-based models (cnn/rnn/lstm/transformer/mlp/dcn/resnet/graph).
    max_grad_norm: Optional[float] = 5.0
    early_stopping_patience: int = 10
    random_state: int = 42
    device_preference: str = "cuda"
    scaler: Optional[str] = "standard"
    min_cells_per_gene: int = 100
    min_expression: float = 0.0
    log1p_transform: bool = False
    target_scaler: Optional[str] = "standard"
    force_target_scaling: bool = False
    enable_smoothing: bool = True
    smoothing_k: int = 20
    smoothing_pca_components: int = 10
    pseudobulk_group_size: int = 1
    pseudobulk_pca_components: int = 10

    min_expression_fraction: float = 0.10
    catboost_iterations: Optional[int] = 1000
    rf_n_estimators: Optional[int] = None
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: Optional[int] = None
    rf_max_features: float | str | None = None
    rf_bootstrap: Optional[bool] = None
    svr_kernel: str = "linear"
    svr_C: float = 1.0
    svr_epsilon: float = 0.1
    svr_max_iter: int = 50_000
    svr_tol: float = 1e-4
    track_history: bool = True
    history_metrics: List[str] = field(
        default_factory=lambda: ["pearson", "r2", "spearman", "rmse", "mse", "mae"]
    )
    group_key: Optional[str] = "sample"
    atac_layer: Optional[str] = "tfidf"
    rna_expression_layer: Optional[str] = "log1p_cpm"
    resource_sample_seconds: float = 60.0
    multioutput_feature_basis: str = "bin"

    # Feature importance configuration for torch models in multi-output mode
    # Enabled by default to always record FI; set samples=None to use all available samples
    enable_feature_importance: bool = False
    feature_importance_samples: Optional[int] = None
    feature_importance_batch_size: int = 256
    # Off by default to avoid generating a large number of panel images.
    enable_per_gene_panels: bool = False
    enable_shap: bool = False
    shap_max_samples: Optional[int] = 500
    shap_background_samples: int = 100
    export_raw_predictions: bool = False
    # When False, keep sparse feature matrices during cell-wise prep to reduce memory.
    # Note: smoothing/pseudobulk or non-sparse scalers will still force densification.
    force_dense_features: bool = True
    # Optional speed profile for heavy classical models in multi-output runs.
    # When enabled, training applies a lighter CV setup and model-specific
    # conservative defaults to reduce wall-time and timeout risk.
    fast_classical_mode: bool = False

    def validate(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Train/val/test fractions must sum to 1.0")
        if self.k_folds < 2:
            raise ValueError("k_folds must be at least 2")
        if self.window_bp <= 0 or self.bin_size_bp <= 0:
            raise ValueError("window_bp and bin_size_bp must be positive")
        if self.effective_batch_cap <= 0:
            raise ValueError("effective_batch_cap must be positive")
        if self.smoothing_k < 1:
            raise ValueError("smoothing_k must be >= 1")
        if self.smoothing_pca_components < 1:
            raise ValueError("smoothing_pca_components must be >= 1")
        if self.pseudobulk_group_size < 1:
            raise ValueError("pseudobulk_group_size must be >= 1")
        if self.pseudobulk_pca_components < 1:
            raise ValueError("pseudobulk_pca_components must be >= 1")
        if self.multioutput_feature_basis not in {"bin", "peak"}:
            raise ValueError("multioutput_feature_basis must be 'bin' or 'peak'")
        if not (0.0 <= self.min_expression_fraction <= 1.0):
            raise ValueError("min_expression_fraction must be within [0, 1]")
        if self.rf_n_estimators is not None and self.rf_n_estimators <= 0:
            raise ValueError("rf_n_estimators must be positive when specified")
        if self.rf_max_depth is not None and self.rf_max_depth <= 0:
            raise ValueError("rf_max_depth must be positive when specified")
        if self.rf_min_samples_leaf is not None and self.rf_min_samples_leaf <= 0:
            raise ValueError("rf_min_samples_leaf must be positive when specified")
        if isinstance(self.rf_max_features, float) and not (0.0 < self.rf_max_features <= 1.0):
            raise ValueError("rf_max_features as a float must be within (0, 1]")
        if self.svr_C <= 0:
            raise ValueError("svr_C must be positive")
        if self.svr_epsilon < 0:
            raise ValueError("svr_epsilon must be non-negative")
        if self.svr_max_iter <= 0:
            raise ValueError("svr_max_iter must be positive")
        if self.svr_tol <= 0:
            raise ValueError("svr_tol must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when specified")
        if self.lr_scheduler not in {"none", "cosine"}:
            raise ValueError("lr_scheduler must be 'none' or 'cosine'")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError("warmup_ratio must be within [0, 1]")
        if not (0.0 < self.min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be within (0, 1]")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.transformer_embed_dim <= 0:
            raise ValueError("transformer_embed_dim must be positive")
        if self.transformer_num_layers <= 0:
            raise ValueError("transformer_num_layers must be positive")
        if not (0.0 <= self.transformer_dropout < 1.0):
            raise ValueError("transformer_dropout must be within [0, 1)")
        if self.transformer_num_heads is not None:
            if self.transformer_num_heads <= 0:
                raise ValueError("transformer_num_heads must be positive when specified")
            if self.transformer_embed_dim % self.transformer_num_heads != 0:
                raise ValueError("transformer_embed_dim must be divisible by transformer_num_heads")
        if self.transformer_arch not in {"v1", "v2"}:
            raise ValueError("transformer_arch must be 'v1' or 'v2'")
        if self.resnet_attention not in {"none", "se"}:
            raise ValueError("resnet_attention must be 'none' or 'se'")
        if self.resnet_attention_se_reduction <= 0:
            raise ValueError("resnet_attention_se_reduction must be positive")
        if self.torch_pearson_loss_weight < 0.0:
            raise ValueError("torch_pearson_loss_weight must be >= 0")
        if self.k_folds > 1 and self.group_key is not None and not self.group_key:
            raise ValueError("group_key must be a non-empty string when provided")
        if self.enable_feature_importance:
            if self.feature_importance_samples is not None and self.feature_importance_samples <= 0:
                raise ValueError("feature_importance_samples must be positive when specified")
            if self.feature_importance_batch_size <= 0:
                raise ValueError("feature_importance_batch_size must be positive")
        if self.enable_shap:
            if self.shap_max_samples is not None and self.shap_max_samples <= 0:
                raise ValueError("shap_max_samples must be positive when specified")
            if self.shap_background_samples <= 0:
                raise ValueError("shap_background_samples must be positive")
        if self.resource_sample_seconds <= 0:
            raise ValueError("resource_sample_seconds must be positive")


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "SPEAR"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    group: Optional[str] = None
    job_type: Optional[str] = None
    log_code: bool = True
    log_dataset_manifest: bool = True
    log_artifacts: bool = True
    log_tables: bool = True
    log_media: bool = True
    log_predictions_table: bool = True
    sweep_overrides: bool = False
    table_max_rows: int = 5000
    media_max_items: int = 50


@dataclass
class ModelConfig:
    model_names: List[str] = field(
        default_factory=lambda: [
            "cnn",
            "rnn",
            "lstm",
            "mlp",
            "xgboost",
            "random_forest",
        ]
    )
    extra_models: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    paths: PathsConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    genes: Optional[List[str]] = None
    chromosomes: Optional[List[str]] = None
    max_genes: Optional[int] = None
    chunk_total: int = 1
    chunk_index: int = 0
    # Default to cell-wise multi-output unless explicitly turned off
    multi_output: bool = True
    run_name: Optional[str] = None
    run_context: Optional[Dict[str, Optional[str]]] = None
    log_path: Optional[Path] = None
    cache_dir: Optional[Path] = None


    def ensure_directories(self) -> None:
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    def all_models(self) -> List[str]:
        return list(dict.fromkeys(self.models.model_names + self.models.extra_models))

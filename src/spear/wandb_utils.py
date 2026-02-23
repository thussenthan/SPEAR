from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional

from .config import PipelineConfig, WandbConfig

_LOG = logging.getLogger(__name__)

try:  # optional W&B dependency
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def _netrc_has_wandb() -> bool:
    netrc_path = Path(os.environ.get("NETRC", str(Path.home() / ".netrc")))
    if not netrc_path.exists():
        return False
    try:
        contents = netrc_path.read_text(errors="ignore")
    except Exception:
        return False
    return "api.wandb.ai" in contents or "wandb.ai" in contents


def _has_api_key_env() -> bool:
    api_key = os.getenv("WANDB_API_KEY")
    return bool(api_key and api_key.strip())



def _wandb_disabled_by_env() -> bool:
    if os.getenv("WANDB_DISABLED"):
        return True
    mode = os.getenv("WANDB_MODE")
    return mode is not None and mode.strip().lower() == "disabled"


def _try_convert_to_scalar(value: Any) -> Any:
    """Convert array-like values (e.g., NumPy/Torch tensors) to Python scalars.

    Args:
        value: The value to convert. May be a scalar, ``None``, or an array-like
            object exposing an ``item()`` method.

    Returns:
        The extracted scalar value when possible; otherwise, the original value.
        Values that are already scalars or ``None`` are returned unchanged. This
        is used to prepare values for JSON serialization in Weights & Biases
        logging.
    """
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, (str, bool, int, float)):
        return value
    return value


def _clean_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            cleaned[key] = _clean_payload(value)
        elif isinstance(value, list):
            cleaned[key] = [_try_convert_to_scalar(v) for v in value]
        else:
            cleaned[key] = _try_convert_to_scalar(value)
    return cleaned


_DATASET_PATTERN = re.compile(r"(embryonic|endothelial)", re.IGNORECASE)
_MODEL_SPECIFIC_TRAINING_PREFIXES = (
    "catboost_",
    "svr_",
    "transformer_",
    "rf_",
)
_MODEL_SPECIFIC_TRAINING_KEYS = {
    "catboost_iterations",
}


def _split_model_specific_training_fields(
    training_payload: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split training payload into generic and model-specific fields."""

    generic: Dict[str, Any] = {}
    model_specific: Dict[str, Any] = {}
    for key, value in training_payload.items():
        is_model_specific = key in _MODEL_SPECIFIC_TRAINING_KEYS or key.startswith(
            _MODEL_SPECIFIC_TRAINING_PREFIXES
        )
        if is_model_specific:
            model_specific[key] = value
        else:
            generic[key] = value
    return generic, model_specific


def _infer_dataset_name(config: PipelineConfig) -> Optional[str]:
    explicit = getattr(config, "dataset", None)
    if explicit:
        return str(explicit).strip().lower()
    paths = config.paths
    candidates = [
        str(paths.base_dir),
        str(paths.atac_path),
        str(paths.rna_path),
        str(paths.gtf_path),
    ]
    for candidate in candidates:
        match = _DATASET_PATTERN.search(candidate)
        if match:
            return match.group(1).lower()
    return None


def infer_dataset_name(config: PipelineConfig) -> Optional[str]:
    return _infer_dataset_name(config)


def _build_wandb_config_payload(config: PipelineConfig) -> Dict[str, Any]:
    training_payload = asdict(config.training)
    payload = {
        "dataset": infer_dataset_name(config),
        "max_genes": config.max_genes,
        "num_requested_genes": len(config.genes) if config.genes else None,
        "chromosomes": config.chromosomes,
        "model": (config.all_models()[0] if config.all_models() else None),
        "training": training_payload,
    }
    # Remove noisy training fields from W&B config to avoid redundancy.
    payload["training"].pop("track_history", None)
    payload["training"].pop("history_metrics", None)
    payload["training"].pop("resource_sample_seconds", None)
    payload["training"].pop("enable_per_gene_panels", None)
    # During sweeps, W&B exposes tuned keys under parameters.*.
    # Drop mirrored training fields to avoid duplicate UI columns.
    if config.wandb.sweep_overrides:
        for key in (
            "gradient_accumulation_steps",
            "lr_scheduler",
            "min_lr_ratio",
            "warmup_epochs",
            "transformer_num_layers",
            "transformer_embed_dim",
            "transformer_dropout",
        ):
            payload["training"].pop(key, None)

    generic_training, model_specific_training = _split_model_specific_training_fields(payload["training"])
    payload["training"] = generic_training
    for key, value in model_specific_training.items():
        payload[f"|training.{key}"] = value

    if config.chunk_total > 1 or config.chunk_index > 0:
        payload["chunk_index"] = config.chunk_index
        payload["chunk_total"] = config.chunk_total

    repro_cmd = os.getenv("SPEAR_WANDB_REPRO_CMD")
    if repro_cmd:
        payload["repro_command"] = repro_cmd

    return _clean_payload(payload)


def _default_wandb_run_name(config_payload: Dict[str, Any]) -> str:
    model = config_payload.get("model") or "model"
    dataset = config_payload.get("dataset") or "unknown"
    gene_count = config_payload.get("max_genes")
    if gene_count is None:
        gene_count = config_payload.get("num_requested_genes")
    gene_label = f"{gene_count}genes" if gene_count not in (None, 0) else "allgenes"
    return f"{model}_{gene_label}_{dataset}"




def maybe_init_wandb(config: PipelineConfig) -> Optional[Any]:
    wandb_cfg: WandbConfig = config.wandb
    if not wandb_cfg.enabled:
        return None
    if _wandb_disabled_by_env():
        _LOG.warning("W&B logging disabled by environment; skipping.")
        return None

    if wandb is None:
        _LOG.warning("W&B enabled but 'wandb' is not installed; skipping.")
        return None

    if not (_has_api_key_env() or _netrc_has_wandb()):
        _LOG.warning("W&B enabled but no API key found (WANDB_API_KEY or ~/.netrc); skipping.")
        return None

    try:
        api_key_raw = os.getenv("WANDB_API_KEY")
        api_key = api_key_raw.strip() if api_key_raw else None
        if api_key:
            _LOG.info("W&B login: attempting WANDB_API_KEY authentication.")
            try:
                wandb.login(key=api_key, relogin=True)
            except Exception as exc:
                if _netrc_has_wandb():
                    _LOG.warning(
                        "W&B login with WANDB_API_KEY failed, ignoring key and retrying with ~/.netrc; error=%s",
                        exc,
                    )
                    # Fall back to credentials from ~/.netrc
                    _LOG.info("W&B login: attempting ~/.netrc authentication.")
                    wandb.login(relogin=True)
                else:
                    # No netrc credentials to fall back to; let the outer handler log and skip
                    raise
        else:
            # Rely on credentials from ~/.netrc, already checked by _netrc_has_wandb()
            _LOG.info("W&B login: attempting ~/.netrc authentication.")
            wandb.login(relogin=True)
    except Exception as exc:
        _LOG.warning("W&B login failed; skipping. error=%s", exc)
        return None

    config_payload = _build_wandb_config_payload(config)
    if not wandb_cfg.run_name:
        wandb_cfg.run_name = _default_wandb_run_name(config_payload)

    try:
        run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=wandb_cfg.run_name or config.run_name,
            group=wandb_cfg.group,
            job_type=wandb_cfg.job_type,
            tags=wandb_cfg.tags if wandb_cfg.tags else None,
            config=config_payload,
        )
        _LOG.info("W&B run initialized | project=%s | name=%s", wandb_cfg.project, run.name)
        if wandb_cfg.log_code:
            try:
                run.log_code(str(config.paths.base_dir))
            except Exception:
                _LOG.debug("W&B code logging failed", exc_info=True)
        return run
    except Exception as exc:
        _LOG.warning("W&B initialization failed; skipping. error=%s", exc)
        return None


def apply_sweep_overrides(config: PipelineConfig, run: Optional[Any]) -> bool:
    """
    Apply W&B sweep configuration overrides to the given pipeline config.

    This function reads parameters from ``run.config`` (if available) and
    mutates the provided ``config`` object in place, updating matching
    training, model, or pipeline-level fields when corresponding keys are
    present in the sweep configuration.

    Returns:
        bool: True if at least one override was successfully applied to
        ``config``, False otherwise (including when no run is provided or
        sweep overrides are disabled).
    """
    if run is None:
        return False
    if not config.wandb.sweep_overrides:
        return False
    try:
        sweep_payload = dict(run.config)
    except Exception as exc:
        _LOG.warning(
            "Failed to read wandb run.config for sweep overrides (%s: %s). "
            "Sweep parameters will be ignored. Please verify your W&B connection "
            "and that the sweep configuration is correctly defined.",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return False
    if not sweep_payload:
        return False

    applied = False
    training_fields = set(config.training.__dict__.keys())
    model_fields = set(config.models.__dict__.keys())
    pipeline_fields = {
        "genes",
        "chromosomes",
        "max_genes",
        "chunk_total",
    }

    training_overrides = sweep_payload.get("training")
    if isinstance(training_overrides, dict):
        for key, value in training_overrides.items():
            if key in training_fields:
                setattr(config.training, key, value)
                applied = True

    models_overrides = sweep_payload.get("models")
    if isinstance(models_overrides, dict):
        for key, value in models_overrides.items():
            if key in model_fields:
                setattr(config.models, key, value)
                applied = True

    for key, value in sweep_payload.items():
        if key in training_fields:
            setattr(config.training, key, value)
            applied = True
        elif key in model_fields:
            setattr(config.models, key, value)
            applied = True
        elif key == "models" and isinstance(value, list):
            config.models.model_names = value
            applied = True
        elif key == "model_names" and isinstance(value, list):
            config.models.model_names = value
            applied = True
        elif key in pipeline_fields:
            setattr(config, key, value)
            applied = True

    if applied:
        try:
            config.training.validate()
        except Exception as exc:
            _LOG.warning("Sweep overrides applied but config validation failed: %s", exc)
            raise
        _LOG.info("Applied sweep overrides from wandb.config")
    return applied


def wandb_log_metrics(run: Optional[Any], metrics: Dict[str, Any], *, step: Optional[int] = None) -> None:
    if run is None:
        return
    try:
        payload = _clean_payload(metrics)
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=step)
    except Exception:
        _LOG.debug("W&B metric logging failed", exc_info=True)


def wandb_update_summary(run: Optional[Any], summary: Dict[str, Any]) -> None:
    if run is None:
        return
    try:
        run.summary.update(_clean_payload(summary))
    except Exception:
        _LOG.debug("W&B summary update failed", exc_info=True)


def wandb_update_config(
    run: Optional[Any],
    config_updates: Dict[str, Any],
    *,
    allow_val_change: bool = True,
) -> None:
    if run is None:
        return
    try:
        run.config.update(_clean_payload(config_updates), allow_val_change=allow_val_change)
    except Exception:
        _LOG.debug("W&B config update failed", exc_info=True)


def wandb_finish(run: Optional[Any], *, status: str, run_dir: Optional[Path] = None) -> None:
    if run is None:
        return
    try:
        status_norm = (status or "").strip().lower()
        summary_payload: Dict[str, Any] = {}
        if run_dir is not None:
            summary_payload["output_dir"] = str(run_dir)
        if summary_payload:
            run.summary.update(_clean_payload(summary_payload))
        exit_code = 0 if status_norm in {"succeeded", "success"} else 1
        run.finish(exit_code=exit_code)
    except Exception:
        _LOG.debug("W&B finish failed", exc_info=True)


def _iter_files_from_globs(run_dir: Path, patterns: Iterable[str]) -> Iterable[Path]:
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            if path.is_file():
                yield path


def log_run_artifacts(run: Optional[Any], run_dir: Path, *, include: Optional[Iterable[str]] = None) -> None:
    if run is None:
        return
    include_patterns = list(include or [])
    if not include_patterns:
        include_patterns = [
            "run_configuration.json",
            "dataset_manifest.json",
            "summary_metrics.csv",
            "summary_metrics_per_gene.csv",
            "selected_genes.csv",
            "models/*/metrics_aggregate.csv",
            "models/*/metrics_per_gene.csv",
            "models/*/metrics_by_gene.csv",
            "models/*/metrics_summary.csv",
            "models/*/metrics_cv.csv",
            "models/*/training_history.csv",
            "models/*/histories/*.csv",
            "models/*/model_meta.json",
            "models/*/feature_scaler.pkl",
            "models/*/model.pt",
            "models/*/feature_importances_mean.csv",
            "models/*/feature_importances_per_gene.csv",
            "models/*/feature_importance_per_gene_summary.csv",
            "models/*/shapley_values_per_gene_summary.csv",
            "models/*/feature_importance_summary.json",
            "models/*/shapley_values_mean.csv",
            "models/*/shapley_values_summary.json",
            "models/*/per_gene_panels/*.png",
        ]

    if wandb is None:
        _LOG.debug("wandb is not installed; skipping artifact logging")
        return

    run_id = getattr(run, "id", None)
    if not run_id:
        _LOG.debug("W&B run object has no 'id' attribute; skipping artifact logging")
        return

    artifact = wandb.Artifact(name=f"spear_run_{run_id}", type="pipeline_results")
    added = 0
    for path in _iter_files_from_globs(run_dir, include_patterns):
        try:
            artifact.add_file(str(path), name=str(path.relative_to(run_dir)))
            added += 1
        except Exception:
            _LOG.debug("Failed to add artifact file %s", path, exc_info=True)

    if added == 0:
        _LOG.debug("No artifact files matched for W&B logging")
        return
    try:
        run.log_artifact(artifact)
        _LOG.info("Logged W&B artifact with %d files", added)
    except Exception:
        _LOG.debug("Failed to log W&B artifact", exc_info=True)


def log_tables_from_csv(
    run: Optional[Any],
    table_name: str,
    csv_path: Path,
    *,
    max_rows: int,
) -> None:
    if run is None or not csv_path.exists():
        return
    if wandb is None:
        return
    try:
        import pandas as pd
    except Exception:
        _LOG.debug("pandas unavailable for W&B table logging", exc_info=True)
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        _LOG.debug("Failed to read CSV for W&B table logging: %s", csv_path, exc_info=True)
        return

    if max_rows > 0 and len(df) > max_rows:
        df = df.head(max_rows)
    try:
        run.log({table_name: wandb.Table(dataframe=df)})
    except Exception:
        _LOG.debug("Failed to log W&B table %s", table_name, exc_info=True)


def log_training_history_charts_from_csv(
    run: Optional[Any],
    csv_path: Path,
    *,
    prefix: str,
) -> None:
    if run is None:
        return
    if wandb is None:
        return
    if not csv_path.exists():
        return
    try:
        import pandas as pd
    except Exception:
        _LOG.debug("pandas unavailable for W&B history charts", exc_info=True)
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        _LOG.debug("Failed to read CSV for W&B history charts: %s", csv_path, exc_info=True)
        return
    if df.empty or "epoch" not in df.columns:
        return

    metric_groups = {
        "Loss": ("train_loss", "val_loss"),
        "Pearson": ("train_pearson", "val_pearson"),
        "R2": ("train_r2", "val_r2"),
        "Spearman": ("train_spearman", "val_spearman"),
        "RMSE": ("train_rmse", "val_rmse"),
        "MSE": ("train_mse", "val_mse"),
        "MAE": ("train_mae", "val_mae"),
    }

    try:
        table = wandb.Table(dataframe=df)
    except Exception:
        _LOG.debug("Failed to build W&B table for history charts: %s", csv_path, exc_info=True)
        return

    for label, (train_col, val_col) in metric_groups.items():
        y_cols = [c for c in (train_col, val_col) if c in df.columns]
        if not y_cols:
            continue
        try:
            chart = wandb.plot.line_series(
                table,
                x="epoch",
                y=y_cols,
                title=f"{prefix} {label}",
            )
            run.log({f"Charts/{prefix}/{label}": chart})
        except Exception:
            _LOG.debug("Failed to log W&B history chart %s (%s)", label, csv_path, exc_info=True)


def log_images_from_globs(
    run: Optional[Any],
    run_dir: Path,
    *,
    patterns: Iterable[str],
    max_items: int,
    prefix: Optional[str] = None,
    group_key: Optional[str] = None,
) -> None:
    if run is None or max_items == 0:
        return
    if wandb is None:
        return

    logged = 0
    grouped_images = [] if group_key else None
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            name = str(path.relative_to(run_dir))
            key = f"{prefix}/{name}" if prefix else name
            try:
                if group_key:
                    grouped_images.append(wandb.Image(str(path), caption=name))
                else:
                    run.log({key: wandb.Image(str(path))})
                logged += 1
            except Exception:
                _LOG.debug("Failed to log W&B image %s", path, exc_info=True)
            if max_items > 0 and logged >= max_items:
                break
        if max_items > 0 and logged >= max_items:
            break
    if group_key and grouped_images:
        try:
            run.log({group_key: grouped_images})
        except Exception:
            _LOG.debug("Failed to log grouped W&B images for %s", group_key, exc_info=True)

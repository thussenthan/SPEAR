from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence

from .config import PipelineConfig, WandbConfig

if TYPE_CHECKING:
    import pandas as pd

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

_METRIC_DISPLAY_NAMES = {
    "pearson": "Pearson",
    "spearman": "Spearman",
    "r2": "R2",
    "rmse": "RMSE",
    "mse": "MSE",
    "mae": "MAE",
}

_METRIC_AXIS_LABELS = {
    "pearson": "Pearson correlation coefficient",
    "spearman": "Spearman correlation coefficient",
    "r2": "R2",
    "rmse": "RMSE",
    "mse": "MSE",
    "mae": "MAE",
}

_METRIC_CHART_PATHS = {
    "pearson": "Pearson",
    "spearman": "Spearman",
    "r2": "R2",
    "rmse": "RMSE",
    "mse": "MSE",
    "mae": "MAE",
}


def _metric_display_name(metric: str) -> str:
    return _METRIC_DISPLAY_NAMES.get(metric, metric.upper())


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


def compute_run_fingerprint(config: PipelineConfig) -> str:
    """Stable hash that uniquely identifies a training run configuration.

    Used to detect duplicate SLURM submissions before any heavy work starts.
    The fingerprint covers all fields that would produce identical results if
    matched: model set, dataset, gene manifest, key hyperparameters, split seed,
    window geometry, and chunk index.
    """
    training = config.training
    payload = {
        "models": sorted(config.all_models()),
        "dataset": infer_dataset_name(config),
        "gene_manifest": str(config.gene_manifest_path or ""),
        "genes_hash": hashlib.sha1(
            ",".join(sorted(config.genes or [])).encode()
        ).hexdigest()[:16],
        "window_bp": training.window_bp,
        "bin_size_bp": training.bin_size_bp,
        "random_state": training.random_state,
        "chunk_index": config.chunk_index,
        "chunk_total": config.chunk_total,
        "multi_output": config.multi_output,
        "optimizer": getattr(training, "optimizer", "adamw"),
        "per_gene_feature_basis": getattr(training, "per_gene_feature_basis", "bin"),
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(raw).hexdigest()


def check_duplicate_run(config: PipelineConfig, fingerprint: str) -> Optional[str]:
    """Query W&B for a completed run with the same fingerprint.

    Returns the URL of the matching run if one exists, or None. All errors
    (network, auth, missing wandb) are caught and logged as warnings so they
    never block a legitimate run.
    """
    if wandb is None:
        return None
    try:
        api = wandb.Api(timeout=15)
        project = config.wandb.project or "SPEAR_v2"
        entity = config.wandb.entity or os.getenv("WANDB_ENTITY") or ""
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(
            path,
            filters={
                "config.run_fingerprint": fingerprint,
                "state": {"$in": ["finished", "running"]},
            },
            per_page=1,
        )
        for run in runs:
            return str(run.url)
    except Exception as exc:
        _LOG.debug("W&B duplicate-check failed (non-fatal): %s", exc)
    return None


def infer_dataset_name(config: PipelineConfig) -> Optional[str]:
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
    # Fall back to the base_dir folder name so new datasets don't log as None
    base_name = Path(str(paths.base_dir)).name
    if base_name and base_name not in {".", "~", ""}:
        return base_name.lower()
    return None


def _build_wandb_config_payload(config: PipelineConfig) -> Dict[str, Any]:
    training_payload = asdict(config.training)
    run_context = config.run_context or {}
    slurm_job_id = run_context.get("slurm_job") or os.getenv("SLURM_JOB_ID")
    slurm_array_task_id = run_context.get("slurm_task") or os.getenv(
        "SLURM_ARRAY_TASK_ID"
    )
    payload = {
        "dataset": infer_dataset_name(config),
        "max_genes": config.max_genes,
        "chromosomes": config.chromosomes,
        "model": (config.all_models()[0] if config.all_models() else None),
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "training": training_payload,
    }
    if config.genes:
        payload["num_requested_genes"] = len(config.genes)
    if config.gene_manifest_path is not None:
        payload["gene_manifest"] = str(config.gene_manifest_path)
    # Remove noisy training fields from W&B config to avoid redundancy.
    payload["training"].pop("track_history", None)
    payload["training"].pop("history_metrics", None)
    payload["training"].pop("resource_sample_seconds", None)
    payload["training"].pop("enable_per_gene_panels", None)
    # These remain preserved in local run configuration exports/artifacts; omit them
    # from W&B config to keep the run overview comparison-focused.
    for key in (
        "atac_layer",
        "rna_expression_layer",
        "device_preference",
        "fast_classical_mode",
    ):
        payload["training"].pop(key, None)
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

    generic_training, model_specific_training = _split_model_specific_training_fields(
        payload["training"]
    )
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


def _single_wandb_model(config: PipelineConfig) -> Optional[str]:
    models = config.all_models()
    if len(models) != 1:
        _LOG.error(
            "W&B logging requires exactly one model per run; requested models=%s.",
            models,
        )
        return None
    return str(models[0])


def maybe_init_wandb(config: PipelineConfig) -> Optional[Any]:
    wandb_cfg: WandbConfig = config.wandb
    if not wandb_cfg.enabled:
        return None
    if _single_wandb_model(config) is None:
        raise ValueError(
            "W&B logging requires exactly one model per run. "
            "Run one model per W&B/Slurm task or disable --wandb."
        )
    if _wandb_disabled_by_env():
        _LOG.warning("W&B logging disabled by environment; skipping.")
        return None

    if wandb is None:
        _LOG.warning("W&B enabled but 'wandb' is not installed; skipping.")
        return None

    if not (_has_api_key_env() or _netrc_has_wandb()):
        _LOG.warning(
            "W&B enabled but no API key found (WANDB_API_KEY or ~/.netrc); skipping."
        )
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
    config_payload["run_fingerprint"] = compute_run_fingerprint(config)
    if not wandb_cfg.run_name:
        wandb_cfg.run_name = _default_wandb_run_name(config_payload)

    try:
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=wandb_cfg.run_name or config.run_name,
            group=wandb_cfg.group,
            job_type=wandb_cfg.job_type,
            tags=wandb_cfg.tags if wandb_cfg.tags else None,
            config=config_payload,
            settings=wandb.Settings(start_method="thread"),
        )
        _LOG.info(
            "W&B run initialized | project=%s | name=%s", wandb_cfg.project, run.name
        )
        if wandb_cfg.log_code:
            try:
                _log_code_snapshot(run, config.paths.base_dir)
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
            if key == "model_names":
                if isinstance(value, list):
                    config.models.model_names = [str(model) for model in value]
                elif isinstance(value, str):
                    config.models.model_names = [value]
                else:
                    config.models.model_names = list(value)
                applied = True
            elif key in model_fields:
                setattr(config.models, key, value)
                applied = True

    for key, value in sweep_payload.items():
        if key in training_fields:
            setattr(config.training, key, value)
            applied = True
        elif key == "model_names":
            if isinstance(value, list):
                config.models.model_names = [str(model) for model in value]
                applied = True
            elif isinstance(value, str):
                config.models.model_names = [value]
                applied = True
        elif key == "models" and isinstance(value, list):
            config.models.model_names = [str(model) for model in value]
            applied = True
        elif key == "model" and isinstance(value, str):
            config.models.model_names = [value]
            applied = True
        elif key in model_fields:
            setattr(config.models, key, value)
            applied = True
        elif key in pipeline_fields:
            setattr(config, key, value)
            applied = True

    if applied:
        try:
            config.training.validate()
        except Exception as exc:
            _LOG.warning(
                "Sweep overrides applied but config validation failed: %s", exc
            )
            raise
        _LOG.info("Applied sweep overrides from wandb.config")
    return applied


def wandb_log_metrics(
    run: Optional[Any], metrics: Dict[str, Any], *, step: Optional[int] = None
) -> None:
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
        run.config.update(
            _clean_payload(config_updates), allow_val_change=allow_val_change
        )
    except Exception:
        _LOG.debug("W&B config update failed", exc_info=True)


def model_used_fallback(model_meta: Dict[str, Any]) -> bool:
    """Return True when a model reports any fallback usage."""
    if model_meta.get("used_fallback"):
        return True
    fallback_reasons = model_meta.get("fallback_reasons")
    if isinstance(fallback_reasons, list) and fallback_reasons:
        return True
    fallbacks = model_meta.get("fallbacks")
    if isinstance(fallbacks, list) and fallbacks:
        return True
    return False


def run_has_fallbacks(model_run_details: Dict[str, Dict[str, Any]]) -> bool:
    """Return True when any model in a run used a fallback path."""
    return any(model_used_fallback(meta) for meta in model_run_details.values())


def wandb_finish(
    run: Optional[Any], *, status: str, run_dir: Optional[Path] = None
) -> None:
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
    seen: set[Path] = set()
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def log_run_artifacts(
    run: Optional[Any], run_dir: Path, *, include: Optional[Iterable[str]] = None
) -> None:
    if run is None:
        return
    include_patterns = list(include or [])
    if not include_patterns:
        include_patterns = [
            "*.json",
            "*.csv",
            "models/*/*.csv",
            "models/*/*.json",
            "models/*/*.pkl",
            "models/*/*.pt",
            "models/*/*.png",
            "models/*/*resource_usage.csv",
            "models/*/*resource_usage.png",
            "models/*/histories/*.csv",
            "models/*/histories/*.png",
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
        _LOG.debug(
            "Failed to read CSV for W&B table logging: %s", csv_path, exc_info=True
        )
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
        _LOG.debug(
            "Failed to read CSV for W&B history charts: %s", csv_path, exc_info=True
        )
        return
    if df.empty or "epoch" not in df.columns:
        return

    available_metrics = []
    for col in df.columns:
        if not col.startswith(("train_", "val_")):
            continue
        metric_name = col.split("_", 1)[1].strip().lower()
        if metric_name and metric_name not in available_metrics:
            available_metrics.append(metric_name)

    preferred_order = ["loss", "pearson", "spearman", "r2", "rmse", "mse", "mae"]
    ordered_metrics = [
        metric for metric in preferred_order if metric in available_metrics
    ]
    ordered_metrics.extend(
        metric for metric in available_metrics if metric not in ordered_metrics
    )

    for metric_name in ordered_metrics:
        train_col = f"train_{metric_name}"
        val_col = f"val_{metric_name}"
        label = "Loss" if metric_name == "loss" else _metric_display_name(metric_name)
        xs = []
        ys = []
        keys = []
        for col in (train_col, val_col):
            if col not in df.columns:
                continue
            aligned = df[["epoch", col]].copy()
            aligned["epoch"] = pd.to_numeric(aligned["epoch"], errors="coerce")
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
            aligned = aligned.dropna(subset=["epoch", col])
            if aligned.empty:
                continue
            xs.append(aligned["epoch"].tolist())
            ys.append(aligned[col].tolist())
            keys.append(col)
        if not xs or not ys:
            continue
        try:
            chart = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=keys,
                title=f"{prefix} {label}",
                xname="epoch",
            )
            run.log({f"Charts/{prefix}/{label}": chart})
        except Exception:
            _LOG.debug(
                "Failed to log W&B history chart %s (%s)",
                label,
                csv_path,
                exc_info=True,
            )


def log_prediction_charts_from_csv(
    run: Optional[Any],
    csv_path: Path,
    *,
    prefix: str,
) -> None:
    if not csv_path.exists():
        return
    df = _load_csv_for_wandb(csv_path)
    if df is None:
        return
    log_prediction_charts_from_dataframe(run, df, prefix=prefix)


def log_prediction_charts_from_dataframe(
    run: Optional[Any],
    df: "pd.DataFrame",
    *,
    prefix: str,
) -> None:
    if run is None:
        return
    if wandb is None:
        return
    try:
        import pandas as pd
    except Exception:
        _LOG.debug("pandas unavailable for W&B prediction charts", exc_info=True)
        return
    if df.empty or "y_true" not in df.columns or "y_pred" not in df.columns:
        return

    numeric = df[["y_true", "y_pred"]].apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.empty:
        return

    if "split" in df.columns:
        split_df = df[["split", "y_true", "y_pred"]].copy()
        split_df["y_true"] = pd.to_numeric(split_df["y_true"], errors="coerce")
        split_df["y_pred"] = pd.to_numeric(split_df["y_pred"], errors="coerce")
        split_df = split_df.dropna(subset=["split", "y_true", "y_pred"])
        if not split_df.empty:
            mean_rows = []
            for split_name, split_rows in split_df.groupby("split"):
                if split_rows.empty:
                    continue
                chart_split = str(split_name).title()
                try:
                    table = wandb.Table(dataframe=split_rows[["y_true", "y_pred"]])
                    chart = wandb.plot.scatter(
                        table,
                        x="y_true",
                        y="y_pred",
                        title=f"{prefix} | {chart_split} Predicted vs Actual",
                    )
                    run.log({f"Charts/Scatter/{chart_split}": chart})
                except Exception:
                    _LOG.debug("Failed to log W&B split scatter chart", exc_info=True)

                residuals = (split_rows["y_pred"] - split_rows["y_true"]).dropna()
                if residuals.empty:
                    continue
                try:
                    res_table = wandb.Table(
                        dataframe=pd.DataFrame({"residual": residuals})
                    )
                    res_chart = wandb.plot.histogram(
                        res_table,
                        "residual",
                        title=f"{prefix} | {chart_split} Residual Distribution",
                    )
                    run.log({f"Charts/Residuals/{chart_split}": res_chart})
                except Exception:
                    _LOG.debug("Failed to log W&B residual histogram", exc_info=True)
                mean_rows.append(
                    {"split": chart_split, "mean_residual": float(residuals.mean())}
                )
            if mean_rows:
                mean_df = pd.DataFrame(mean_rows)
                numeric = pd.to_numeric(
                    mean_df["mean_residual"], errors="coerce"
                ).dropna()
                if not numeric.empty and numeric.nunique(dropna=True) > 1:
                    try:
                        mean_chart = wandb.plot.bar(
                            wandb.Table(dataframe=mean_df),
                            "split",
                            "mean_residual",
                            title=f"{prefix} | Mean Residual by Split",
                        )
                        run.log({"Charts/Residual_Bar": mean_chart})
                    except Exception:
                        _LOG.debug(
                            "Failed to log W&B mean residual bar chart", exc_info=True
                        )


def log_metric_distribution_charts_from_csv(
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
        _LOG.debug("pandas unavailable for W&B metric charts", exc_info=True)
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        _LOG.debug(
            "Failed to read CSV for W&B metric charts: %s", csv_path, exc_info=True
        )
        return
    if df.empty:
        return

    metrics = ("pearson", "r2", "spearman", "rmse", "mse", "mae")

    # --- Compute generalization gap from the raw df before normalizing format ---
    gap_series: Optional[Any] = None
    if "split" in df.columns and "gene" in df.columns and "pearson" in df.columns:
        metric_frame = df.loc[:, ["gene", "split", "pearson"]].copy()
        metric_frame["gene"] = metric_frame["gene"].astype(str)
        metric_frame["pearson"] = pd.to_numeric(
            metric_frame["pearson"], errors="coerce"
        )
        metric_frame = metric_frame.dropna(subset=["gene", "split", "pearson"])
        if not metric_frame.empty:
            pivot = metric_frame.pivot_table(
                index="gene", columns="split", values="pearson", aggfunc="mean"
            )
            if "train" in pivot.columns and "test" in pivot.columns:
                gap_series = (pivot["train"] - pivot["test"]).dropna()
    elif "train_pearson" in df.columns and "test_pearson" in df.columns:
        train_vals = pd.to_numeric(df["train_pearson"], errors="coerce")
        test_vals = pd.to_numeric(df["test_pearson"], errors="coerce")
        mask = train_vals.notna() & test_vals.notna()
        if mask.any():
            gap_series = (train_vals[mask] - test_vals[mask]).dropna()

    # --- Normalize both CSV formats into a single long DataFrame ---
    # Long format: has a "split" column
    # Wide format: has columns like "train_pearson", "test_pearson", etc.
    if "split" in df.columns:
        long_df = df.copy()
        long_df["split"] = long_df["split"].astype(str).str.lower().str.title()
    else:
        rows = []
        for split in ("train", "val", "test"):
            for metric in metrics:
                col = f"{split}_{metric}"
                if col not in df.columns:
                    continue
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                for v in values:
                    rows.append({"split": split.title(), metric: float(v)})
        long_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if long_df.empty:
        return

    # --- Single chart-building pass over all metrics ---
    for metric in metrics:
        if metric not in long_df.columns:
            continue
        split_distribution_df = pd.DataFrame(
            {
                "split": long_df["split"],
                "value": pd.to_numeric(long_df[metric], errors="coerce"),
            }
        ).dropna(subset=["split", "value"])
        if split_distribution_df.empty:
            continue
        fig = _build_plotly_split_distribution_figure(
            split_distribution_df,
            split_col="split",
            value_col="value",
            title=f"{prefix} | {_metric_display_name(metric)} by Split",
            yaxis_title=_METRIC_AXIS_LABELS.get(metric, metric.upper()),
        )
        if fig is not None:
            try:
                run.log(
                    {f"Charts/{_METRIC_CHART_PATHS.get(metric, metric.upper())}": fig}
                )
            except Exception:
                _LOG.debug(
                    "Failed to log W&B Plotly split distribution for %s",
                    metric,
                    exc_info=True,
                )
        else:
            for split_label in ("Train", "Val", "Test"):
                values = split_distribution_df.loc[
                    split_distribution_df["split"] == split_label, "value"
                ]
                if values.empty:
                    continue
                try:
                    hist_table = wandb.Table(dataframe=pd.DataFrame({"value": values}))
                    hist_chart = wandb.plot.histogram(
                        hist_table,
                        "value",
                        title=f"{prefix} | {split_label} {_metric_display_name(metric)} Distribution",
                    )
                    run.log(
                        {
                            f"Charts/{_METRIC_CHART_PATHS.get(metric, metric.upper())}/{split_label}": hist_chart
                        }
                    )
                except Exception:
                    _LOG.debug(
                        "Failed to log W&B metric histogram fallback for %s",
                        metric,
                        exc_info=True,
                    )

    # --- Generalization gap chart ---
    if gap_series is not None and not gap_series.empty:
        gap_df = pd.DataFrame({"value": gap_series})
        fig = _build_plotly_gap_distribution_figure(
            gap_df["value"],
            title=f"{prefix} | Train-Test Pearson Gap",
            yaxis_title="Train - Test Pearson",
        )
        if fig is not None:
            try:
                run.log({"Charts/Generalization_Gap": fig})
            except Exception:
                _LOG.debug(
                    "Failed to log W&B Plotly generalization gap chart (%s)",
                    csv_path,
                    exc_info=True,
                )
        else:
            try:
                gap_table = wandb.Table(dataframe=gap_df)
                gap_chart = wandb.plot.histogram(
                    gap_table,
                    "value",
                    title=f"{prefix} | Train-Test Pearson Gap",
                )
                run.log({"Charts/Generalization_Gap": gap_chart})
            except Exception:
                _LOG.debug(
                    "Failed to log W&B generalization gap histogram fallback (%s)",
                    csv_path,
                    exc_info=True,
                )


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
            _LOG.debug(
                "Failed to log grouped W&B images for %s", group_key, exc_info=True
            )


def _load_csv_for_wandb(csv_path: Path) -> Optional["pd.DataFrame"]:
    try:
        import pandas as pd
    except Exception:
        _LOG.debug("pandas unavailable for W&B CSV logging", exc_info=True)
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        _LOG.debug("Failed to read CSV for W&B logging: %s", csv_path, exc_info=True)
        return None


def _log_table(run: Optional[Any], key: str, df: "pd.DataFrame") -> None:
    if run is None or wandb is None or df.empty:
        return
    try:
        run.log({key: wandb.Table(dataframe=df)})
    except Exception:
        _LOG.debug("Failed to log W&B table %s", key, exc_info=True)


def _make_numeric_frame(df: "pd.DataFrame", columns: Sequence[str]) -> "pd.DataFrame":
    import pandas as pd

    out = df.loc[:, [col for col in columns if col in df.columns]].copy()
    for col in out.columns:
        if col == "feature":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([float("inf"), float("-inf")], pd.NA)


def _build_plotly_split_distribution_figure(
    df: "pd.DataFrame",
    *,
    split_col: str,
    value_col: str,
    title: str,
    yaxis_title: str,
) -> Optional[Any]:
    try:
        import plotly.graph_objects as go
    except Exception:
        _LOG.debug(
            "plotly unavailable for W&B custom split distribution plot", exc_info=True
        )
        return None

    split_values = df[split_col].astype(str).tolist()
    ordered_splits = [
        split for split in ("Train", "Val", "Test") if split in set(split_values)
    ]
    if not ordered_splits:
        ordered_splits = list(dict.fromkeys(split_values))

    palette = {"Train": "#4C72B0", "Val": "#DD8452", "Test": "#55A868"}
    fig = go.Figure()
    for split in ordered_splits:
        split_df = df[df[split_col].astype(str) == split]
        if split_df.empty:
            continue
        fig.add_trace(
            go.Violin(
                x=[split] * len(split_df),
                y=split_df[value_col],
                name=split,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.12,
                pointpos=0.0,
                marker={"size": 4, "opacity": 0.28},
                line={"color": palette.get(split, "#4C72B0")},
                fillcolor=palette.get(split, "#4C72B0"),
                opacity=0.55,
            )
        )
    fig.update_layout(
        title=title,
        violinmode="group",
        xaxis_title="Split",
        yaxis_title=yaxis_title,
        legend_title_text="Split",
        template="plotly_white",
    )
    return fig


def _build_plotly_gap_distribution_figure(
    values: "pd.Series",
    *,
    title: str,
    yaxis_title: str,
) -> Optional[Any]:
    try:
        import plotly.graph_objects as go
    except Exception:
        _LOG.debug("plotly unavailable for W&B custom gap plot", exc_info=True)
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            x=["Train-Test"] * len(values),
            y=values,
            name="Train-Test",
            box_visible=True,
            meanline_visible=True,
            points="all",
            jitter=0.12,
            pointpos=0.0,
            marker={"size": 4, "opacity": 0.28},
            line={"color": "#C44E52"},
            fillcolor="#C44E52",
            opacity=0.6,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Gap",
        yaxis_title=yaxis_title,
        showlegend=False,
        template="plotly_white",
    )
    return fig


def log_feature_summary_charts_from_csv(
    run: Optional[Any],
    csv_path: Path,
    *,
    prefix: str,
    value_col: str,
    title_prefix: str,
    signed_col: Optional[str] = None,
    distance_col: str = "signed_distance_to_tss_kb",
    top_n: int = 30,
) -> None:
    if run is None or wandb is None or not csv_path.exists():
        return
    df = _load_csv_for_wandb(csv_path)
    if (
        df is None
        or df.empty
        or "feature" not in df.columns
        or value_col not in df.columns
    ):
        return

    selected_columns = ["feature", value_col, distance_col]
    if signed_col:
        selected_columns.append(signed_col)
    plot_df = _make_numeric_frame(df, selected_columns)
    plot_df["feature"] = df["feature"].astype(str)
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        return

    plot_df["_abs_value"] = plot_df[value_col].abs()
    top_df = (
        plot_df.sort_values("_abs_value", ascending=False).head(max(1, top_n)).copy()
    )
    if not top_df.empty:
        try:
            chart = wandb.plot.bar(
                wandb.Table(dataframe=top_df[["feature", value_col]]),
                "feature",
                value_col,
                title=f"{title_prefix} | Top {len(top_df)}",
            )
            run.log({f"Charts/{prefix}": chart})
        except Exception:
            _LOG.debug(
                "Failed to log W&B top-feature chart from %s", csv_path, exc_info=True
            )

    if signed_col and signed_col in plot_df.columns:
        signed_df = plot_df.dropna(subset=[signed_col]).copy()
        if not signed_df.empty:
            top_pos = signed_df.sort_values(signed_col, ascending=False).head(10)
            top_neg = signed_df.sort_values(signed_col, ascending=True).head(10)
            import pandas as pd

            signed_top = pd.concat(
                [top_pos, top_neg], ignore_index=True
            ).drop_duplicates(subset=["feature"])
            signed_top = signed_top.sort_values(signed_col, ascending=True)
            try:
                chart = wandb.plot.bar(
                    wandb.Table(dataframe=signed_top[["feature", signed_col]]),
                    "feature",
                    signed_col,
                    title=f"{title_prefix} | Signed",
                )
                run.log({f"Charts/{prefix}/Signed": chart})
            except Exception:
                _LOG.debug(
                    "Failed to log W&B signed-feature chart from %s",
                    csv_path,
                    exc_info=True,
                )

    if distance_col in plot_df.columns:
        dist_df = plot_df.dropna(subset=[distance_col]).copy()
        if not dist_df.empty:
            dist_df = dist_df.rename(columns={distance_col: "distance_kb"})
            try:
                scatter = wandb.plot.scatter(
                    wandb.Table(dataframe=dist_df[["distance_kb", value_col]]),
                    x="distance_kb",
                    y=value_col,
                    title=f"{title_prefix} vs TSS distance",
                )
                run.log({f"Charts/{prefix}/TSS_Distance": scatter})
            except Exception:
                _LOG.debug(
                    "Failed to log W&B distance scatter from %s",
                    csv_path,
                    exc_info=True,
                )

            dist_profile = dist_df.copy()
            dist_profile["abs_distance_kb"] = dist_profile["distance_kb"].abs()
            dist_profile["_weight"] = dist_profile[value_col].abs()
            dist_profile = dist_profile.sort_values("abs_distance_kb")
            total_weight = float(dist_profile["_weight"].sum())
            if total_weight > 0.0:
                dist_profile["cumulative_fraction"] = (
                    dist_profile["_weight"].cumsum() / total_weight
                )
            else:
                dist_profile["cumulative_fraction"] = 0.0
            dist_profile = dist_profile.groupby("abs_distance_kb", as_index=False)[
                "cumulative_fraction"
            ].max()
            try:
                line = wandb.plot.line_series(
                    xs=[dist_profile["abs_distance_kb"].tolist()],
                    ys=[dist_profile["cumulative_fraction"].tolist()],
                    keys=["cumulative_fraction"],
                    title=f"{title_prefix} cumulative distance profile",
                    xname="abs_distance_kb",
                )
                run.log({f"Charts/{prefix}/Cumulative_Overview": line})
            except Exception:
                _LOG.debug(
                    "Failed to log W&B cumulative distance chart from %s",
                    csv_path,
                    exc_info=True,
                )


def log_resource_usage_charts_from_csv(
    run: Optional[Any],
    csv_path: Path,
    *,
    prefix: str = "Resources",
) -> None:
    if run is None or wandb is None or not csv_path.exists():
        return
    df = _load_csv_for_wandb(csv_path)
    if df is None or df.empty or "time_sec" not in df.columns:
        return
    import pandas as pd

    df = df.copy()
    numeric_cols = [
        "rss_gib",
        "cpu_percent",
        "thread_count",
        "gpu_memory_gib",
        "gpu_reserved_gib",
        "gpu_util_percent",
    ]
    for col in ["time_sec", *numeric_cols]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["time_sec"])
    if df.empty:
        return

    chart_sets = {
        "Memory": ["rss_gib", "gpu_memory_gib", "gpu_reserved_gib"],
        "Utilization": ["cpu_percent", "gpu_util_percent"],
        "Threads": ["thread_count"],
    }
    for label, cols in chart_sets.items():
        y_cols = [col for col in cols if col in df.columns and df[col].notna().any()]
        varied_cols = []
        for col in y_cols:
            numeric = pd.to_numeric(df[col], errors="coerce").dropna()
            if numeric.empty or numeric.nunique(dropna=True) <= 1:
                continue
            varied_cols.append(col)
        y_cols = varied_cols
        if not y_cols:
            continue
        try:
            chart = wandb.plot.line_series(
                xs=[df["time_sec"].tolist() for _ in y_cols],
                ys=[df[col].tolist() for col in y_cols],
                keys=y_cols,
                title=f"{prefix} | {label}",
                xname="time_sec",
            )
            run.log({f"Charts/{prefix}/{label}": chart})
        except Exception:
            _LOG.debug(
                "Failed to log W&B resource chart %s from %s",
                label,
                csv_path,
                exc_info=True,
            )


def _log_code_snapshot(run: Any, base_dir: Path) -> None:
    base_dir = base_dir.resolve()
    excluded_dir_parts = {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "output",
        "wandb",
    }
    excluded_patterns = (
        "*.h5ad",
        "*.pt",
        "*.pth",
        "*.pkl",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.pdf",
        "*.csv",
        "*.tsv",
        "*.parquet",
        "*.feather",
        "*.npz",
        "*.npy",
    )

    def _exclude(path: str, root: str) -> bool:
        try:
            rel = Path(path).resolve().relative_to(Path(root).resolve())
        except Exception:
            rel = Path(path)
        if any(part in excluded_dir_parts for part in rel.parts):
            return True
        rel_str = rel.as_posix()
        return any(fnmatch.fnmatch(rel_str, pattern) for pattern in excluded_patterns)

    run.log_code(str(base_dir), exclude_fn=_exclude)

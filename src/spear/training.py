
import hashlib
import json
import logging
import random
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import contextlib

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainingConfig
from .data import GeneInfo
from .metrics import regression_metrics
from .models import TorchModelBundle, build_model
from .logging_utils import get_logger, _get_gpu_utilization_percent
from .cache import (
    save_prepared_data,
    load_prepared_data,
    save_prepared_cellwise_data,
    load_prepared_cellwise_data,
)
from .data_types import PreparedData, PreparedCellwiseData, SplitData, CellwiseSplitData

# Suppress specific warnings that are informational only
warnings.filterwarnings('ignore', message='.*CuDNN issue.*nvrtc.so.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Ill-conditioned matrix.*', category=RuntimeWarning)


# Global resource tracker for peak values across entire run
_RESOURCE_TRACKER = {
    "peak_rss_gib": 0.0,
    "peak_cpu_pct": 0.0,
    "peak_gpu_allocated_mb": 0.0,
    "peak_gpu_reserved_mb": 0.0,
    "peak_gpu_free_mb": float("inf"),
    "max_gpu_devices": 0,
}
_CPU_PRIMED = False


def _get_gpu_utilization_pct() -> Optional[float]:
    """Return GPU utilization percentage if available.
    
    Queries GPU utilization via the thread-safe NVML interface from logging_utils.
    Returns None if GPU is unavailable or utilization cannot be determined.
    
    Returns
    -------
    Optional[float]
        GPU utilization as a percentage (0-100), or None if unavailable.
    """
    try:
        return _get_gpu_utilization_percent()
    except Exception:
        return None


def get_resource_summary() -> dict:
    """Return dictionary of peak resource values accumulated during the run."""
    return _RESOURCE_TRACKER.copy()


try:  # psutil is optional; best-effort resource visibility
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:  # optional GPU utilization reporting
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None

class _NoopGradScaler:
    """Minimal stand-in when AMP is disabled or unavailable."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:  # pragma: no cover - trivial
        pass

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:  # pragma: no cover
        pass


if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and hasattr(torch.amp, "autocast"):
    _AMP_GRAD_SCALER = torch.amp.GradScaler
    _AMP_AUTOCAST = torch.amp.autocast
else:  # pragma: no cover - AMP unavailable
    _AMP_GRAD_SCALER = None
    _AMP_AUTOCAST = None


def _make_grad_scaler(use_amp: bool):
    if not use_amp or _AMP_GRAD_SCALER is None:
        return _NoopGradScaler()
    return _AMP_GRAD_SCALER(enabled=True)


def _amp_autocast(device_type: str, use_amp: bool):
    if not use_amp or _AMP_AUTOCAST is None:
        return contextlib.nullcontext()
    return _AMP_AUTOCAST(device_type=device_type, enabled=True)


def _reshape_tensor_for_model(tens: torch.Tensor, reshape: str | None) -> torch.Tensor:
    """Utility to reshape tensor for model input, shared between training and prediction."""
    if reshape == "sequence":
        return tens.reshape(tens.shape[0], -1, 1)
    return tens


def _should_scale_targets(config: TrainingConfig) -> bool:
    """
    Decide whether target values should be scaled based on the training configuration.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration containing options that control target scaling,
        including the choice of target scaler, whether scaling is forced, and
        whether logarithmic transforms are already applied to the targets.

    Returns
    -------
    bool
        True if target scaling should be applied, False if it should be skipped.

    Notes
    -----
    Target scaling is skipped when:

    * ``config.target_scaler`` is ``None`` or ``"none"``.
    * A logarithmic transform is already in use via ``config.log1p_transform`` or
      an RNA expression layer whose name starts with ``"log"``.

    Target scaling is always applied when:

    * ``config.force_target_scaling`` is True, regardless of other settings.
    """
    if config.target_scaler in (None, "none"):
        return False
    if config.force_target_scaling:
        return True
    if config.log1p_transform or (
        config.rna_expression_layer and config.rna_expression_layer.lower().startswith("log")
    ):
        return False
    return True

_LOG = get_logger(__name__)


def _wrap_model_for_multi_gpu(model: nn.Module, device: torch.device) -> nn.Module:
    """Wrap model in DataParallel if multiple GPUs are available and device is CUDA."""
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        _LOG.info("Wrapping model in DataParallel for %d GPUs", torch.cuda.device_count())
        return nn.DataParallel(model)
    return model


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:  # pragma: no cover - best-effort seeding
        _LOG.debug("Failed to apply full torch seeding", exc_info=True)


def _log_resource_snapshot(label: str) -> None:
    global _CPU_PRIMED
    if psutil is None:
        return
    process = psutil.Process()
    try:
        rss_bytes = process.memory_info().rss
        rss_gib = rss_bytes / (1024 ** 3)
        _RESOURCE_TRACKER["peak_rss_gib"] = max(_RESOURCE_TRACKER["peak_rss_gib"], rss_gib)
    except Exception:  # pragma: no cover - defensive fallback
        rss_gib = float("nan")
    try:
        if not _CPU_PRIMED:
            # Prime the CPU percent sampler and use the priming sample as the first value.
            cpu_pct = process.cpu_percent(interval=0.1)
            _CPU_PRIMED = True
        else:
            cpu_pct = process.cpu_percent(interval=None)
        _RESOURCE_TRACKER["peak_cpu_pct"] = max(_RESOURCE_TRACKER["peak_cpu_pct"], cpu_pct)
    except Exception:  # pragma: no cover - defensive fallback
        cpu_pct = float("nan")
    _LOG.info(
        "Resource snapshot | %s | rss=%.2f GiB | cpu%%=%.1f",
        label,
        rss_gib,
        cpu_pct,
    )


def _log_gpu_memory_snapshot(label: str) -> None:
    """Log GPU memory usage if CUDA is available.
    
    Captures:
        - Reserved: Total GPU memory allocated by PyTorch
        - Allocated: Currently in-use GPU memory
        - Cached: Memory held by caching allocator (available for reuse)
        - Free: Unallocated device memory
    
    Tracks peak values globally for final summary.
    """
    if not torch.cuda.is_available():
        return
    
    try:

        torch.cuda.synchronize()
        
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        
        # Peak memory since last reset
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # Available on device
        device_count = torch.cuda.device_count()
        total_device_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        free_device_mb = total_device_mb - reserved_mb
        
        # Update tracker
        _RESOURCE_TRACKER["peak_gpu_allocated_mb"] = max(_RESOURCE_TRACKER["peak_gpu_allocated_mb"], peak_allocated_mb)
        _RESOURCE_TRACKER["peak_gpu_reserved_mb"] = max(_RESOURCE_TRACKER["peak_gpu_reserved_mb"], reserved_mb)
        _RESOURCE_TRACKER["peak_gpu_free_mb"] = min(_RESOURCE_TRACKER["peak_gpu_free_mb"], free_device_mb)
        _RESOURCE_TRACKER["max_gpu_devices"] = max(_RESOURCE_TRACKER["max_gpu_devices"], device_count)
        
        _LOG.info(
            "GPU memory snapshot | %s | allocated=%.0f MB (peak=%.0f MB) | reserved=%.0f MB | "
            "free=%.0f MB / %.0f MB total | devices=%d",
            label,
            allocated_mb,
            peak_allocated_mb,
            reserved_mb,
            free_device_mb,
            total_device_mb,
            device_count,
        )
    except Exception:  # pragma: no cover - defensive fallback
        _LOG.debug("Failed to capture GPU memory snapshot", exc_info=True)


def _config_cache_key(config: TrainingConfig, scope: str) -> str:
    payload = {
        "scope": scope,
        "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
        "test_fraction": config.test_fraction,
        "random_state": config.random_state,
        "group_key": config.group_key,
        "enable_smoothing": config.enable_smoothing,
        "smoothing_k": config.smoothing_k,
        "smoothing_pca_components": config.smoothing_pca_components,
        "pseudobulk_group_size": config.pseudobulk_group_size,
        "pseudobulk_pca_components": config.pseudobulk_pca_components,
        "scaler": config.scaler or "none",
        "target_scaler": config.target_scaler or "none",
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass
class FoldMetrics:
    fold: int
    metrics: Dict[str, float]


@dataclass
class ModelResult:
    """
    Results of a per-gene model trained for a single gene.

    Attributes
    ----------
    gene_name
        Gene identifier (e.g., gene symbol or ID).
    model_name
        Name of the model used.
    cv_metrics
        Cross-validation metrics across folds.
    train_metrics
        Training set performance metrics.
    val_metrics
        Validation set performance metrics.
    test_metrics
        Test set performance metrics.
    predictions
        Structured array containing predictions and ground truth values.
    fitted_model
        The fitted model object (optional).
    history
        Training history records (optional).
    """
    gene_name: str
    model_name: str
    cv_metrics: List[FoldMetrics]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    predictions: np.recarray
    fitted_model: Optional[object] = None
    history: Optional[List[Dict[str, float]]] = None


@dataclass
class CellwiseModelResult:
    model_name: str
    gene_names: List[str]
    gene_infos: Optional[List[GeneInfo]]
    cv_metrics: List[FoldMetrics]
    aggregate_metrics: Dict[str, Dict[str, float]]
    per_gene_metrics: Dict[str, List[Dict[str, float]]]
    split_predictions: Dict[str, Dict[str, np.ndarray]]
    fitted_model: Optional[object] = None
    history: Optional[List[Dict[str, float]]] = None
    feature_importances: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    feature_importance_method: Optional[str] = None
    shap_importances_mean: Optional[np.ndarray] = None
    shap_importance_method: Optional[str] = None
    feature_block_slices: Optional[List[Tuple[int, int]]] = None
    feature_block_indices: Optional[List[np.ndarray]] = None
    feature_scaler: Optional[StandardScaler | MinMaxScaler] = None
    target_scaler: Optional[StandardScaler | MinMaxScaler] = None
    reshape: Optional[str] = None


def prepare_data(dataset, config: TrainingConfig, cache_dir: Optional[Path] = None) -> PreparedData:
    cache_key = _config_cache_key(config, "gene")
    cache_dict = getattr(dataset, "prepared_cache", {})
    if cache_dict is None:
        cache_dict = {}
        setattr(dataset, "prepared_cache", cache_dict)

    # Try in-memory cache first
    if cache_key in cache_dict:
        return cache_dict[cache_key]  # type: ignore[return-value]
    
    # Try disk cache if directory provided
    if cache_dir is not None:
        disk_cached = load_prepared_data(cache_dir, cache_key)
        if disk_cached is not None:
            # Restore to in-memory cache too
            cache_dict[cache_key] = disk_cached
            return disk_cached

    _seed_everything(config.random_state)

    _prep_start = time.perf_counter()
    _log_resource_snapshot("prepare_data:start")
    # Use genes[0].gene_name if available, else fallback to 'unknown'.
    if hasattr(dataset, "genes") and dataset.genes and hasattr(dataset.genes[0], "gene_name"):
        gene_name = dataset.genes[0].gene_name
    else:
        try:
            gene_name = dataset.gene.gene_name
        except Exception:
            gene_name = "unknown"
    _LOG.info("Preparing dataset for gene %s with %d cells", gene_name, dataset.num_cells())

    X = dataset.X.astype(np.float32)
    y = dataset.y.astype(np.float32)
    cells = dataset.cell_ids
    groups = getattr(dataset, "group_labels", None)
    if groups is None:
        groups = np.asarray(cells)
    else:
        groups = np.asarray(groups)

    use_group_split = bool(config.group_key)
    if use_group_split:
        unique_groups = np.unique(groups)
        if unique_groups.size < 2:
            _LOG.warning(
                "Grouped splitting requested but only %d unique groups found; falling back to random split",
                unique_groups.size,
            )
            use_group_split = False
    rng_state = config.random_state

    if use_group_split:
        splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=rng_state)
        try:
            train_val_idx, test_idx = next(splitter.split(X, y, groups))
        except ValueError:
            _LOG.warning("Falling back to random train/test split due to insufficient groups")
            use_group_split = False
        else:
            X_temp = X[train_val_idx]
            y_temp = y[train_val_idx]
            cell_temp = cells[train_val_idx]
            group_temp = groups[train_val_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            cell_test = cells[test_idx]
            group_test = groups[test_idx]

    if not use_group_split:
        X_temp, X_test, y_temp, y_test, cell_temp, cell_test, group_temp, group_test = train_test_split(
            X,
            y,
            cells,
            groups,
            test_size=config.test_fraction,
            random_state=rng_state,
        )
        group_temp = np.asarray(group_temp)
        group_test = np.asarray(group_test)

    val_ratio = config.val_fraction / (config.train_fraction + config.val_fraction)

    if use_group_split:
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=rng_state + 1)
        try:
            train_idx_rel, val_idx_rel = next(val_splitter.split(X_temp, y_temp, group_temp))
        except ValueError:
            _LOG.warning("Falling back to random train/val split due to insufficient groups")
            use_group_split = False
        else:
            X_train = X_temp[train_idx_rel]
            y_train = y_temp[train_idx_rel]
            cell_train = cell_temp[train_idx_rel]
            group_train = group_temp[train_idx_rel]
            X_val = X_temp[val_idx_rel]
            y_val = y_temp[val_idx_rel]
            cell_val = cell_temp[val_idx_rel]
            group_val = group_temp[val_idx_rel]

    if not use_group_split:
        X_train, X_val, y_train, y_val, cell_train, cell_val, group_train, group_val = train_test_split(
            X_temp,
            y_temp,
            cell_temp,
            group_temp,
            test_size=val_ratio,
            random_state=rng_state + 1,
        )
        group_train = np.asarray(group_train)
        group_val = np.asarray(group_val)

    if config.enable_smoothing and config.smoothing_k > 1:
        X_train, y_train, cell_train = _apply_knn_smoothing(
            X_train,
            y_train,
            cell_train,
            group_size=config.smoothing_k,
            n_components=config.smoothing_pca_components,
            random_state=config.random_state,
            split_label="train",
        )
        X_val, y_val, cell_val = _apply_knn_smoothing(
            X_val,
            y_val,
            cell_val,
            group_size=config.smoothing_k,
            n_components=config.smoothing_pca_components,
            random_state=config.random_state + 1,
            split_label="val",
        )
        X_test, y_test, cell_test = _apply_knn_smoothing(
            X_test,
            y_test,
            cell_test,
            group_size=config.smoothing_k,
            n_components=config.smoothing_pca_components,
            random_state=config.random_state + 2,
            split_label="test",
        )

    X_train, y_train, cell_train, group_train = _apply_pseudobulk(
        X_train,
        y_train,
        cell_train,
        group_labels=group_train,
        group_size=config.pseudobulk_group_size,
        n_components=config.pseudobulk_pca_components,
        random_state=config.random_state,
        split_label="train",
    )
    X_val, y_val, cell_val, group_val = _apply_pseudobulk(
        X_val,
        y_val,
        cell_val,
        group_labels=group_val,
        group_size=config.pseudobulk_group_size,
        n_components=config.pseudobulk_pca_components,
        random_state=config.random_state + 1,
        split_label="val",
    )
    X_test, y_test, cell_test, group_test = _apply_pseudobulk(
        X_test,
        y_test,
        cell_test,
        group_labels=group_test,
        group_size=config.pseudobulk_group_size,
        n_components=config.pseudobulk_pca_components,
        random_state=config.random_state + 2,
        split_label="test",
    )

    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()
    y_train_raw = y_train.copy()
    y_val_raw = y_val.copy()
    y_test_raw = y_test.copy()

    feature_scaler: Optional[StandardScaler | MinMaxScaler]
    if config.scaler == "standard":
        feature_scaler = StandardScaler()
    elif config.scaler == "minmax":
        feature_scaler = MinMaxScaler()
    else:
        feature_scaler = None

    if feature_scaler is not None:
        X_train = feature_scaler.fit_transform(X_train)
        X_val = feature_scaler.transform(X_val)
        X_test = feature_scaler.transform(X_test)

    log_targets = bool(config.log1p_transform) or (
        config.rna_expression_layer and "log" in config.rna_expression_layer.lower()
    )
    skip_target_scaling = (
        log_targets
        and not getattr(config, "force_target_scaling", False)
        and config.target_scaler in {"standard", "minmax"}
    )

    target_scaler: Optional[StandardScaler | MinMaxScaler]
    if skip_target_scaling:
        _LOG.info("Skipping target scaling because targets are already log-transformed")
        target_scaler = None
    elif config.target_scaler == "standard":
        target_scaler = StandardScaler()
    elif config.target_scaler == "minmax":
        target_scaler = MinMaxScaler()
    else:
        target_scaler = None

    if target_scaler is not None:
        y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    splits = SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        cell_ids_train=cell_train,
        cell_ids_val=cell_val,
        cell_ids_test=cell_test,
        group_labels_train=group_train,
        group_labels_val=group_val,
        group_labels_test=group_test,
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        y_train_raw=y_train_raw,
        y_val_raw=y_val_raw,
        y_test_raw=y_test_raw,
    )
    _LOG.info(
        "Prepared gene-wise splits | train=%d | val=%d | test=%d | features=%d | %.2fs",
        X_train.shape[0],
        X_val.shape[0],
        X_test.shape[0],
        X_train.shape[1],
        time.perf_counter() - _prep_start,
    )
    _log_resource_snapshot("prepare_data:end")
    
    prepared_data = PreparedData(splits=splits, feature_scaler=feature_scaler, target_scaler=target_scaler)
    
    # Save to disk cache if directory provided
    if cache_dir is not None:
        try:
            save_prepared_data(prepared_data, cache_dir, cache_key)
        except Exception as exc:
            _LOG.warning("Failed to save prepared data to disk cache: %s", exc)
    
    cache_dict[cache_key] = prepared_data
    return prepared_data


def _cellwise_cache_key(dataset, config: TrainingConfig) -> str:
    cache_key = _config_cache_key(config, "cellwise")
    # Make cache dataset-specific to avoid collisions across datasets with same config.
    try:
        sig = f"{dataset.X.shape}-{dataset.y.shape}"
    except Exception:
        # Fallback: build a stable identifier from common dataset attributes (if available)
        parts = []
        for attr in ("name", "id", "path", "file_path", "filename"):
            if hasattr(dataset, attr):
                try:
                    value = getattr(dataset, attr)
                except Exception:
                    continue
                parts.append(f"{attr}={value}")
        if parts:
            sig = "|".join(parts)
        else:
            # Last-resort: fall back to the dataset's type name, which is stable across runs
            sig = f"unknown-{type(dataset).__name__}"
    return f"{cache_key}_{hashlib.sha256(sig.encode('utf-8')).hexdigest()}"


def prepare_cellwise_data(dataset, config: TrainingConfig, cache_dir: Optional[Path] = None) -> PreparedCellwiseData:
    cache_key = _cellwise_cache_key(dataset, config)
    cache_dict = getattr(dataset, "prepared_cache", {})
    if cache_dict is None:
        cache_dict = {}
        setattr(dataset, "prepared_cache", cache_dict)
    
    # Try in-memory cache first
    if cache_key in cache_dict:
        return cache_dict[cache_key]  # type: ignore[return-value]
    
    # Try disk cache if directory provided
    if cache_dir is not None:
        disk_cached = load_prepared_cellwise_data(cache_dir, cache_key)
        if disk_cached is not None:
            # Restore to in-memory cache too
            cache_dict[cache_key] = disk_cached
            return disk_cached
    
    _LOG.info(
        "Preparing cell-wise data | cells=%d | features=%d | targets=%d",
        dataset.X.shape[0],
        dataset.X.shape[1],
        dataset.y.shape[1] if dataset.y.ndim > 1 else 1,
    )
    _log_resource_snapshot("prepare_cellwise_data:start")
    _cellwise_prep_start = time.perf_counter()

    X = dataset.X
    force_dense = getattr(config, "force_dense_features", True)
    needs_dense = bool(config.enable_smoothing and config.smoothing_k > 1) or config.pseudobulk_group_size > 1
    if sp.issparse(X) and not force_dense and needs_dense:
        _LOG.warning(
            "Sparse features with smoothing/pseudobulk require dense arrays; forcing densification."
        )
        force_dense = True
    if sp.issparse(X) and not force_dense and config.scaler == "minmax":
        _LOG.warning("MinMax scaling does not support sparse inputs; forcing densification.")
        force_dense = True
    # NOTE: Cell-wise training typically expects dense features for downstream scaling/modeling.
    # For large sparse ATAC matrices, this conversion can be memory intensive; keep sparse
    # if allowed by config to avoid OOMs.
    if sp.issparse(X):
        if force_dense:
            # Estimate memory usage of dense float32 matrix before densification.
            n_rows, n_cols = X.shape
            # float32 uses 4 bytes per element.
            estimated_bytes = int(n_rows) * int(n_cols) * 4
            # Warn if the estimated size is larger than 1 GiB.
            one_gib = 1024 ** 3
            if estimated_bytes > one_gib:
                _LOG.warning(
                    "Converting sparse matrix to dense may require approximately %.2f GiB of memory "
                    "(shape=%d x %d). This may lead to out-of-memory errors.",
                    estimated_bytes / one_gib,
                    n_rows,
                    n_cols,
                )
            X = X.toarray().astype(np.float32)
        else:
            X = X.astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
    Y = dataset.y.astype(np.float32)
    cells = dataset.cell_ids
    groups = getattr(dataset, "group_labels", None)
    if groups is None:
        groups = np.asarray(cells)
    else:
        groups = np.asarray(groups)
    rng_state = config.random_state

    use_group_split = bool(config.group_key)
    if use_group_split:
        unique_groups = np.unique(groups)
        if unique_groups.size < 2:
            _LOG.warning(
                "Grouped splitting requested but only %d unique groups found; falling back to random split",
                unique_groups.size,
            )
            use_group_split = False

    if use_group_split:
        splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=rng_state)
        try:
            train_val_idx, test_idx = next(splitter.split(X, Y, groups))
        except ValueError:
            _LOG.warning("Falling back to random train/test split for cellwise data due to insufficient groups")
            use_group_split = False
        else:
            X_temp = X[train_val_idx]
            Y_temp = Y[train_val_idx]
            cell_temp = cells[train_val_idx]
            group_temp = groups[train_val_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]
            cell_test = cells[test_idx]
            group_test = groups[test_idx]

    if not use_group_split:
        X_temp, X_test, Y_temp, Y_test, cell_temp, cell_test, group_temp, group_test = train_test_split(
            X,
            Y,
            cells,
            groups,
            test_size=config.test_fraction,
            random_state=rng_state,
        )
        group_temp = np.asarray(group_temp)
        group_test = np.asarray(group_test)

    val_ratio = config.val_fraction / (config.train_fraction + config.val_fraction)

    if use_group_split:
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=rng_state + 1)
        try:
            train_idx_rel, val_idx_rel = next(val_splitter.split(X_temp, Y_temp, group_temp))
        except ValueError:
            _LOG.warning("Falling back to random train/val split for cellwise data")
            use_group_split = False
        else:
            X_train = X_temp[train_idx_rel]
            Y_train = Y_temp[train_idx_rel]
            cell_train = cell_temp[train_idx_rel]
            group_train = group_temp[train_idx_rel]
            X_val = X_temp[val_idx_rel]
            Y_val = Y_temp[val_idx_rel]
            cell_val = cell_temp[val_idx_rel]
            group_val = group_temp[val_idx_rel]

    if not use_group_split:
        X_train, X_val, Y_train, Y_val, cell_train, cell_val, group_train, group_val = train_test_split(
            X_temp,
            Y_temp,
            cell_temp,
            group_temp,
            test_size=val_ratio,
            random_state=rng_state + 1,
        )
        group_train = np.asarray(group_train)
        group_val = np.asarray(group_val)

    if config.enable_smoothing and config.smoothing_k > 1:
        X_train, Y_train, cell_train = _apply_knn_smoothing(
            X_train,
            Y_train,
            cell_train,
            group_size=config.smoothing_k,
            n_components=config.smoothing_pca_components,
            random_state=config.random_state,
            split_label="train",
        )
        X_val, Y_val, cell_val = _apply_knn_smoothing(
            X_val,
            Y_val,
            cell_val,
            group_size=config.smoothing_k,
            n_components=config.smoothing_pca_components,
            random_state=config.random_state + 1,
            split_label="val",
        )
        X_test, Y_test, cell_test = _apply_knn_smoothing(
            X_test,
            Y_test,
            cell_test,
            group_size=config.smoothing_k,
            n_components=config.smoothing_pca_components,
            random_state=config.random_state + 2,
            split_label="test",
        )

    X_train, Y_train, cell_train, group_train = _apply_pseudobulk(
        X_train,
        Y_train,
        cell_train,
        group_labels=group_train,
        group_size=config.pseudobulk_group_size,
        n_components=config.pseudobulk_pca_components,
        random_state=config.random_state,
        split_label="train",
    )
    X_val, Y_val, cell_val, group_val = _apply_pseudobulk(
        X_val,
        Y_val,
        cell_val,
        group_labels=group_val,
        group_size=config.pseudobulk_group_size,
        n_components=config.pseudobulk_pca_components,
        random_state=config.random_state + 1,
        split_label="val",
    )
    X_test, Y_test, cell_test, group_test = _apply_pseudobulk(
        X_test,
        Y_test,
        cell_test,
        group_labels=group_test,
        group_size=config.pseudobulk_group_size,
        n_components=config.pseudobulk_pca_components,
        random_state=config.random_state + 2,
        split_label="test",
    )

    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()
    Y_train_raw = Y_train.copy()
    Y_val_raw = Y_val.copy()
    Y_test_raw = Y_test.copy()

    feature_scaler: Optional[StandardScaler | MinMaxScaler]
    if config.scaler == "standard":
        feature_scaler = StandardScaler(with_mean=not sp.issparse(X_train))
    elif config.scaler == "minmax":
        feature_scaler = MinMaxScaler()
    else:
        feature_scaler = None

    if feature_scaler is not None:
        X_train = feature_scaler.fit_transform(X_train)
        X_val = feature_scaler.transform(X_val)
        X_test = feature_scaler.transform(X_test)

    log_targets = bool(config.log1p_transform) or (
        config.rna_expression_layer and "log" in config.rna_expression_layer.lower()
    )
    skip_target_scaling = (
        log_targets
        and not getattr(config, "force_target_scaling", False)
        and config.target_scaler in {"standard", "minmax"}
    )

    target_scaler: Optional[StandardScaler | MinMaxScaler]
    if skip_target_scaling:
        _LOG.info("Skipping target scaling because targets are already log-transformed")
        target_scaler = None
    elif config.target_scaler == "standard":
        target_scaler = StandardScaler()
    elif config.target_scaler == "minmax":
        target_scaler = MinMaxScaler()
    else:
        target_scaler = None

    if target_scaler is not None:
        Y_train = target_scaler.fit_transform(Y_train)
        Y_val = target_scaler.transform(Y_val)
        Y_test = target_scaler.transform(Y_test)

    splits = CellwiseSplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=Y_train,
        y_val=Y_val,
        y_test=Y_test,
        cell_ids_train=cell_train,
        cell_ids_val=cell_val,
        cell_ids_test=cell_test,
        group_labels_train=group_train,
        group_labels_val=group_val,
        group_labels_test=group_test,
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        y_train_raw=Y_train_raw,
        y_val_raw=Y_val_raw,
        y_test_raw=Y_test_raw,
    )
    _LOG.info(
        "Prepared cell-wise splits | train=%d | val=%d | test=%d | features=%d | %.2fs",
        X_train.shape[0],
        X_val.shape[0],
        X_test.shape[0],
        X_train.shape[1],
        time.perf_counter() - _cellwise_prep_start,
    )
    _log_resource_snapshot("prepare_cellwise_data:end")
    
    prepared_cellwise = PreparedCellwiseData(splits=splits, feature_scaler=feature_scaler, target_scaler=target_scaler)
    
    # Save to disk cache if directory provided
    if cache_dir is not None:
        try:
            save_prepared_cellwise_data(prepared_cellwise, cache_dir, cache_key)
        except Exception as exc:
            _LOG.warning("Failed to save prepared cell-wise data to disk cache: %s", exc)
    
    cache_dict[cache_key] = prepared_cellwise
    return prepared_cellwise


def train_model_for_gene(
    dataset,
    model_name: str,
    config: TrainingConfig,
    artifacts_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> ModelResult:
    _seed_everything(config.random_state)
    cache_key = _config_cache_key(config, scope="gene")
    cache_dict = getattr(dataset, "prepared_cache", {})
    if cache_dict is None:
        cache_dict = {}
        setattr(dataset, "prepared_cache", cache_dict)
    prepared: PreparedData
    if cache_key in cache_dict:
        prepared = cache_dict[cache_key]  # type: ignore[assignment]
        if hasattr(dataset, "gene"):
            _LOG.info("Reusing cached prepared data for gene %s", getattr(dataset.gene, "gene_name", "unknown"))
    else:
        prepared = prepare_data(dataset, config, cache_dir=cache_dir)
        cache_dict[cache_key] = prepared
    splits = prepared.splits

    cv_groups = np.asarray(splits.group_labels_train)
    unique_groups = np.unique(cv_groups)
    use_group_kfold = bool(config.group_key) and unique_groups.size >= config.k_folds
    if use_group_kfold:
        kf = GroupKFold(n_splits=config.k_folds)
        splitter = kf.split(splits.X_train, splits.y_train, cv_groups)
    else:
        if config.group_key:
            _LOG.warning(
                "Insufficient unique groups (%d) for GroupKFold (k=%d); falling back to standard KFold",
                unique_groups.size,
                config.k_folds,
            )
        kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=config.random_state)
        splitter = kf.split(splits.X_train)
    cv_metrics: List[FoldMetrics] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter, start=1):
        X_train_source = splits.X_train_raw if splits.X_train_raw is not None else splits.X_train
        y_train_source = splits.y_train_raw if splits.y_train_raw is not None else splits.y_train

        X_tr_raw = X_train_source[train_idx]
        X_va_raw = X_train_source[val_idx]
        y_tr_raw = y_train_source[train_idx]
        y_va_raw = y_train_source[val_idx]

        if prepared.feature_scaler is not None and splits.X_train_raw is not None:
            fold_feature_scaler = clone(prepared.feature_scaler)
            X_tr = fold_feature_scaler.fit_transform(X_tr_raw)
            X_va = fold_feature_scaler.transform(X_va_raw)
        else:
            X_tr = X_tr_raw
            X_va = X_va_raw

        fold_target_scaler: Optional[StandardScaler | MinMaxScaler] = None
        if prepared.target_scaler is not None and splits.y_train_raw is not None:
            fold_target_scaler = clone(prepared.target_scaler)
            y_tr_scaled = fold_target_scaler.fit_transform(_ensure_2d(y_tr_raw))
            y_va_scaled = fold_target_scaler.transform(_ensure_2d(y_va_raw))
            if y_tr_raw.ndim == 1:
                y_tr = y_tr_scaled.ravel()
                y_va = y_va_scaled.ravel()
            else:
                y_tr = y_tr_scaled
                y_va = y_va_scaled
        else:
            y_tr = y_tr_raw
            y_va = y_va_raw
        fold_artifacts_dir = None
        if artifacts_dir is not None and model_name == "catboost":
            fold_artifacts_dir = artifacts_dir / f"cv_fold_{fold_idx}"
        model = build_model(
            model_name,
            dataset.X.shape[1],
            config,
            artifacts_dir=fold_artifacts_dir,
        )
        if isinstance(model, TorchModelBundle):
            _, preds, _ = _fit_torch_model(
                model,
                X_tr,
                y_tr,
                X_va,
                y_va,
                config,
                target_scaler=fold_target_scaler,
                capture_history=False,
            )
        else:
            estimator = clone(model)
            estimator.fit(X_tr, y_tr)
            preds = estimator.predict(X_va)
        scaler_for_metrics = fold_target_scaler if fold_target_scaler is not None else prepared.target_scaler
        metrics = regression_metrics(
            _unscale_targets(scaler_for_metrics, y_va),
            _unscale_targets(scaler_for_metrics, preds),
        )
        cv_metrics.append(FoldMetrics(fold=fold_idx, metrics=metrics))
        _LOG.info(
            "CV fold %d | gene=%s | model=%s | R2=%.4f | RMSE=%.4f",
            fold_idx,
            dataset.gene.gene_name,
            model_name,
            metrics.get("r2", float("nan")),
            metrics.get("rmse", float("nan")),
        )

    final_artifacts_dir = None
    if artifacts_dir is not None and model_name == "catboost":
        final_artifacts_dir = artifacts_dir / "final_fit"
    model = build_model(
        model_name,
        dataset.X.shape[1],
        config,
        artifacts_dir=final_artifacts_dir,
    )
    if isinstance(model, TorchModelBundle):
        fitted_model, _, history = _fit_torch_model(
            model,
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            config,
            target_scaler=prepared.target_scaler,
            capture_history=True,
        )
        pred_train = _predict_torch(fitted_model, model.reshape, config, splits.X_train)
        pred_val = _predict_torch(fitted_model, model.reshape, config, splits.X_val)
        pred_test = _predict_torch(fitted_model, model.reshape, config, splits.X_test)
    else:
        estimator = clone(model)
        estimator.fit(splits.X_train, splits.y_train)
        pred_train = estimator.predict(splits.X_train)
        pred_val = estimator.predict(splits.X_val)
        pred_test = estimator.predict(splits.X_test)
        fitted_model = estimator  # type: ignore
        history = None

    y_train_true = _unscale_targets(prepared.target_scaler, splits.y_train)
    y_val_true = _unscale_targets(prepared.target_scaler, splits.y_val)
    y_test_true = _unscale_targets(prepared.target_scaler, splits.y_test)

    train_metrics = regression_metrics(y_train_true, _unscale_targets(prepared.target_scaler, pred_train))
    val_metrics = regression_metrics(y_val_true, _unscale_targets(prepared.target_scaler, pred_val))
    test_metrics = regression_metrics(y_test_true, _unscale_targets(prepared.target_scaler, pred_test))

    _LOG.info(
        "Final metrics | gene=%s | model=%s | train_R2=%.4f | val_R2=%.4f | test_R2=%.4f",
        dataset.gene.gene_name,
        model_name,
        train_metrics.get("r2", float("nan")),
        val_metrics.get("r2", float("nan")),
        test_metrics.get("r2", float("nan")),
    )

    predictions = _stack_predictions(
        dataset.gene.gene_name,
        model_name,
        {
            "train": (splits.cell_ids_train, y_train_true, _unscale_targets(prepared.target_scaler, pred_train)),
            "val": (splits.cell_ids_val, y_val_true, _unscale_targets(prepared.target_scaler, pred_val)),
            "test": (splits.cell_ids_test, y_test_true, _unscale_targets(prepared.target_scaler, pred_test)),
        },
    )

    return ModelResult(
        gene_name=dataset.gene.gene_name,
        model_name=model_name,
        cv_metrics=cv_metrics,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        predictions=predictions,
        fitted_model=fitted_model,
        history=history,
    )


def _compute_torch_feature_importance(
    bundle: TorchModelBundle,
    X_reference: np.ndarray,
    y_reference: Optional[np.ndarray],
    *,
    device: torch.device,
    max_samples: int = 2000,
    batch_size: int = 256,
    target_scaler: Optional[StandardScaler | MinMaxScaler] = None,
) -> Optional[np.ndarray]:
    """Estimate global feature importance via mean absolute input gradients with permutation fallback."""

    X_ref = np.asarray(X_reference, dtype=np.float32)
    y_ref = np.asarray(y_reference, dtype=np.float64) if y_reference is not None else None

    if X_ref.size == 0:
        return None


    sample_limit = max_samples if max_samples is not None and max_samples > 0 else None

    if sample_limit is not None and X_ref.shape[0] > sample_limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_ref.shape[0], size=sample_limit, replace=False)
        idx.sort()
        X_ref = X_ref[idx]
        if y_ref is not None:
            y_ref = y_ref[idx]

    if y_ref is not None and y_ref.shape[0] != X_ref.shape[0]:
        # Align reference targets with the sampled feature matrix to avoid shape mismatches downstream.
        min_len = min(y_ref.shape[0], X_ref.shape[0])
        X_ref = X_ref[:min_len]
        y_ref = y_ref[:min_len]

    model = bundle.model.to(device)
    model = _wrap_model_for_multi_gpu(model, device)
    model.eval()

    totals = np.zeros(X_ref.shape[1], dtype=np.float64)
    count = 0

    grad_success = False
    try:
        for start in range(0, X_ref.shape[0], batch_size):
            batch = X_ref[start : start + batch_size]
            tensor = torch.tensor(batch, device=device, dtype=torch.float32)
            tensor = _reshape_tensor_for_model(tensor, bundle.reshape)
            tensor.requires_grad_(True)

            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outputs = model(tensor)
                loss = outputs.sum()
                loss.backward()

            grads = tensor.grad
            if grads is None:
                continue
            if bundle.reshape == "sequence":
                grads = grads.reshape(grads.shape[0], -1)

            grad_batch = grads.detach().abs().cpu().numpy()
            totals += grad_batch.sum(axis=0)
            count += grad_batch.shape[0]
        if count > 0:
            grad_success = True
            return totals / float(count)
    except Exception:
        grad_success = False

    if not grad_success and y_ref is not None:
        return _compute_torch_permutation_importance(
            bundle,
            X_ref,
            y_ref,
            device=device,
            target_scaler=target_scaler,
            max_samples=max_samples,
            batch_size=batch_size,
        )
    return None


def _compute_torch_shap_importance(
    bundle: TorchModelBundle,
    X_reference: np.ndarray,
    *,
    device: torch.device,
    max_samples: Optional[int] = 500,
    background_samples: int = 100,
) -> Optional[np.ndarray]:
    """Estimate mean absolute SHAP values for a torch model (best-effort)."""
    try:
        import shap
    except Exception as exc:  # pragma: no cover - optional dependency
        _LOG.info("SHAP unavailable; skipping SHAP export (%s).", exc)
        return None

    X_ref = np.asarray(X_reference, dtype=np.float32)
    if X_ref.size == 0:
        return None

    rng = np.random.default_rng(42)
    sample_limit = max_samples if max_samples is not None and max_samples > 0 else None
    if sample_limit is not None and X_ref.shape[0] > sample_limit:
        idx = rng.choice(X_ref.shape[0], size=sample_limit, replace=False)
        idx.sort()
        X_ref = X_ref[idx]

    background_size = min(background_samples, X_ref.shape[0])
    if background_size <= 0:
        return None
    background_idx = rng.choice(X_ref.shape[0], size=background_size, replace=False)
    background = X_ref[background_idx]

    model = bundle.model.to(device)
    model = _wrap_model_for_multi_gpu(model, device)
    model.eval()

    class _ShapWrapper(nn.Module):
        def __init__(self, inner: nn.Module, reshape: Optional[str]) -> None:
            super().__init__()
            self.inner = inner
            self.reshape = reshape

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = _reshape_tensor_for_model(x, self.reshape)
            out = self.inner(x)
            if out.ndim == 1:
                return out.unsqueeze(1)
            if out.ndim == 2:
                if out.shape[1] > 1:
                    return out.mean(dim=1, keepdim=True)
                return out
            out = out.reshape(out.shape[0], -1)
            if out.shape[1] > 1:
                return out.mean(dim=1, keepdim=True)
            return out

    wrapper = _ShapWrapper(model, bundle.reshape)
    background_tensor = torch.tensor(background, device=device, dtype=torch.float32)
    sample_tensor = torch.tensor(X_ref, device=device, dtype=torch.float32)

    try:
        explainer = shap.GradientExplainer(wrapper, background_tensor)
        shap_values = explainer.shap_values(sample_tensor)
    except Exception as exc:
        msg = (
            "Failed to compute SHAP values using GradientExplainer. "
            "This often indicates that the model output shape is incompatible with SHAP's expectations. "
            f"SHAP expects the first dimension of the model output to match the input batch size "
            f"(got input batch size {sample_tensor.shape[0]})."
        )
        with contextlib.suppress(Exception):
            test_out = wrapper(sample_tensor[:1])
            msg += f" Example model output shape for batch_size=1: {tuple(test_out.shape)}."
        _LOG.error("%s Original error: %s", msg, exc)
        return None
    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=0)
        if shap_arr.ndim > 3:
            shap_arr = shap_arr.reshape(shap_arr.shape[0], shap_arr.shape[1], -1)
        if shap_arr.ndim != 3:
            return None
        return np.mean(np.abs(shap_arr), axis=(0, 1))

    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim > 2:
        shap_arr = shap_arr.reshape(shap_arr.shape[0], -1)
    if shap_arr.ndim != 2:
        return None
    return np.mean(np.abs(shap_arr), axis=0)


def _compute_torch_permutation_importance(
    bundle: TorchModelBundle,
    X_reference: np.ndarray,
    y_reference: np.ndarray,
    *,
    device: torch.device,
    target_scaler: Optional[StandardScaler | MinMaxScaler] = None,
    max_samples: int = 500,
    batch_size: int = 256,
) -> Optional[np.ndarray]:
    """Permutation importance on a sample subset using MSE delta."""

    X_ref = np.asarray(X_reference, dtype=np.float32)
    y_ref = np.asarray(y_reference, dtype=np.float64)
    if X_ref.size == 0 or y_ref.size == 0:
        return None

    sample_limit = max_samples if max_samples is not None and max_samples > 0 else None
    if sample_limit is not None and X_ref.shape[0] > sample_limit:
        rng = np.random.default_rng(13)
        idx = rng.choice(X_ref.shape[0], size=sample_limit, replace=False)
        idx.sort()
        X_ref = X_ref[idx]
        y_ref = y_ref[idx]

    def _predict_unscaled(inputs: np.ndarray) -> np.ndarray:
        tens = torch.tensor(inputs, device=device, dtype=torch.float32)
        tens = _reshape_tensor_for_model(tens, bundle.reshape)
        with torch.no_grad():
            model_device = bundle.model.to(device)
            model_device = _wrap_model_for_multi_gpu(model_device, device)
            preds = model_device(tens).cpu().numpy()
        return _unscale_targets(target_scaler, preds)

    base_pred = _predict_unscaled(X_ref)
    base_mse = float(np.mean((base_pred - y_ref) ** 2))

    importances = np.zeros(X_ref.shape[1], dtype=np.float64)
    rng = np.random.default_rng(37)
    for feat_idx in range(X_ref.shape[1]):
        permuted = X_ref.copy()
        rng.shuffle(permuted[:, feat_idx])
        perm_pred = _predict_unscaled(permuted)
        perm_mse = float(np.mean((perm_pred - y_ref) ** 2))
        importances[feat_idx] = max(0.0, perm_mse - base_mse)

    return importances


def train_multi_output_model(
    dataset,
    model_name: str,
    config: TrainingConfig,
    artifacts_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> CellwiseModelResult:
    _seed_everything(config.random_state)
    cache_key = _cellwise_cache_key(dataset, config)
    cache_dict = getattr(dataset, "prepared_cache", {})
    if cache_dict is None:
        cache_dict = {}
        setattr(dataset, "prepared_cache", cache_dict)
    prepared: PreparedCellwiseData
    if cache_key in cache_dict:
        prepared = cache_dict[cache_key]  # type: ignore[assignment]
        _LOG.info(
            "Reusing cached prepared cell-wise data for %d genes",
            len(getattr(dataset, "genes", [])),
        )
    else:
        prepared = prepare_cellwise_data(dataset, config, cache_dir=cache_dir)
        cache_dict[cache_key] = prepared
    splits = prepared.splits
    gene_names = [gene.gene_name for gene in dataset.genes]
    target_dim = dataset.y.shape[1]
    catboost_artifacts_root = artifacts_dir if artifacts_dir is not None and model_name == "catboost" else None

    _LOG.info(
        "Training multi-output model | model=%s | genes=%d | train_samples=%d | features=%d | targets=%d",
        model_name,
        len(gene_names),
        splits.X_train.shape[0],
        splits.X_train.shape[1],
        target_dim,
    )
    _log_resource_snapshot(f"train_multi_output:{model_name}:start")

    cv_groups = np.asarray(splits.group_labels_train)
    unique_groups = np.unique(cv_groups)
    use_group_kfold = bool(config.group_key) and unique_groups.size >= config.k_folds
    if use_group_kfold:
        kf = GroupKFold(n_splits=config.k_folds)
        splitter = kf.split(splits.X_train, splits.y_train, cv_groups)
    else:
        if config.group_key:
            _LOG.warning(
                "Insufficient unique groups (%d) for GroupKFold (k=%d); falling back to KFold",
                unique_groups.size,
                config.k_folds,
            )
        kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=config.random_state)
        splitter = kf.split(splits.X_train)
    cv_metrics: List[FoldMetrics] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter, start=1):
        fold_start = time.perf_counter()
        X_train_source = splits.X_train_raw if splits.X_train_raw is not None else splits.X_train
        y_train_source = splits.y_train_raw if splits.y_train_raw is not None else splits.y_train

        X_tr_raw = X_train_source[train_idx]
        X_va_raw = X_train_source[val_idx]
        y_tr_raw = y_train_source[train_idx]
        y_va_raw = y_train_source[val_idx]

        if prepared.feature_scaler is not None and splits.X_train_raw is not None:
            fold_feature_scaler = clone(prepared.feature_scaler)
            X_tr = fold_feature_scaler.fit_transform(X_tr_raw)
            X_va = fold_feature_scaler.transform(X_va_raw)
        else:
            X_tr = X_tr_raw
            X_va = X_va_raw

        fold_target_scaler: Optional[StandardScaler | MinMaxScaler] = None
        if prepared.target_scaler is not None and splits.y_train_raw is not None:
            fold_target_scaler = clone(prepared.target_scaler)
            y_tr = fold_target_scaler.fit_transform(_ensure_2d(y_tr_raw))
            y_va = fold_target_scaler.transform(_ensure_2d(y_va_raw))
        else:
            y_tr = y_tr_raw
            y_va = y_va_raw
        _LOG.info(
            "Starting CV fold %d/%d | model=%s | train_samples=%d | val_samples=%d",
            fold_idx,
            config.k_folds,
            model_name,
            X_tr.shape[0],
            X_va.shape[0],
        )
        _log_resource_snapshot(f"train_multi_output:{model_name}:fold{fold_idx}:start")
        fold_artifacts_dir = None
        if catboost_artifacts_root is not None:
            fold_artifacts_dir = catboost_artifacts_root / f"cv_fold_{fold_idx}"
        model = build_model(
            model_name,
            dataset.X.shape[1],
            config,
            output_dim=target_dim,
            artifacts_dir=fold_artifacts_dir,
        )
        if isinstance(model, TorchModelBundle):
            _, preds, _ = _fit_torch_model(
                model,
                X_tr,
                y_tr,
                X_va,
                y_va,
                config,
                target_scaler=fold_target_scaler,
                capture_history=False,
            )
        else:
            estimator = clone(model)
            estimator.fit(X_tr, y_tr)
            preds = estimator.predict(X_va)

        scaler_for_metrics = fold_target_scaler if fold_target_scaler is not None else prepared.target_scaler
        y_val_true = _ensure_2d(_unscale_targets(scaler_for_metrics, y_va))
        y_val_pred = _ensure_2d(_unscale_targets(scaler_for_metrics, preds))
        agg_metrics, _ = _compute_multi_metrics(y_val_true, y_val_pred, gene_names)
        cv_metrics.append(FoldMetrics(fold=fold_idx, metrics=agg_metrics))
        duration = time.perf_counter() - fold_start
        _LOG.info(
            "Completed CV fold %d/%d | model=%s | mean_R2=%.4f | mean_RMSE=%.4f | %.2fs",
            fold_idx,
            config.k_folds,
            model_name,
            agg_metrics.get("r2", float("nan")),
            agg_metrics.get("rmse", float("nan")),
            duration,
        )
        _log_resource_snapshot(f"train_multi_output:{model_name}:fold{fold_idx}:end")

    final_artifacts_dir = None
    if catboost_artifacts_root is not None:
        final_artifacts_dir = catboost_artifacts_root / "final_fit"
    model = build_model(
        model_name,
        dataset.X.shape[1],
        config,
        output_dim=target_dim,
        artifacts_dir=final_artifacts_dir,
    )
    fit_start = time.perf_counter()
    if isinstance(model, TorchModelBundle):
        fitted_model, _, history = _fit_torch_model(
            model,
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            config,
            target_scaler=prepared.target_scaler,
            capture_history=True,
        )
        pred_train = _predict_torch(fitted_model, model.reshape, config, splits.X_train, output_dim=target_dim)
        pred_val = _predict_torch(fitted_model, model.reshape, config, splits.X_val, output_dim=target_dim)
        pred_test = _predict_torch(fitted_model, model.reshape, config, splits.X_test, output_dim=target_dim)
    else:
        estimator = clone(model)
        estimator.fit(splits.X_train, splits.y_train)
        pred_train = estimator.predict(splits.X_train)
        pred_val = estimator.predict(splits.X_val)
        pred_test = estimator.predict(splits.X_test)
        fitted_model = estimator  # type: ignore
        history = None

    fit_duration = time.perf_counter() - fit_start
    _LOG.info(
        "Completed full fit | model=%s | duration=%.2fs | (exporting outputs...)",
        model_name,
        fit_duration,
    )
    _log_resource_snapshot(f"train_multi_output:{model_name}:fit:end")

    feature_importances: Optional[np.ndarray] = None
    feature_importance_method: Optional[str] = None
    shap_importances_mean: Optional[np.ndarray] = None
    shap_importance_method: Optional[str] = None
    if isinstance(model, TorchModelBundle) and config.enable_feature_importance:
        fi_start = time.perf_counter()
        fi_max_samples = config.feature_importance_samples
        fi_batch_size = config.feature_importance_batch_size
        try:
            device = _select_device(config.device_preference)
            feature_importances = _compute_torch_feature_importance(
                model,
                splits.X_val,
                splits.y_val,
                device=device,
                max_samples=fi_max_samples if fi_max_samples is not None else splits.X_val.shape[0],
                batch_size=fi_batch_size,
                target_scaler=prepared.target_scaler,
            )
            if feature_importances is not None:
                feature_importance_method = "input_gradient_abs_mean"
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            _LOG.warning("Failed to compute feature importances for %s: %s", model_name, exc)
            feature_importances = None
            feature_importance_method = None
        finally:
            fi_elapsed = time.perf_counter() - fi_start
            eff_samples = int(min(splits.X_val.shape[0], fi_max_samples) if fi_max_samples is not None else splits.X_val.shape[0])
            _LOG.info(
                "Feature importance completed | model=%s | method=%s | samples=%d | batch_size=%d | %.2fs",
                model_name,
                feature_importance_method or "n/a",
                eff_samples,
                fi_batch_size,
                fi_elapsed,
            )
    if isinstance(model, TorchModelBundle) and config.enable_shap:
        shap_start = time.perf_counter()
        try:
            device = _select_device(config.device_preference)
            shap_importances_mean = _compute_torch_shap_importance(
                model,
                splits.X_val,
                device=device,
                max_samples=config.shap_max_samples,
                background_samples=config.shap_background_samples,
            )
            if shap_importances_mean is not None:
                shap_importance_method = "shap_gradient_explainer_mean_abs"
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            _LOG.warning("Failed to compute SHAP importances for %s: %s", model_name, exc)
            shap_importances_mean = None
            shap_importance_method = None
        finally:
            shap_elapsed = time.perf_counter() - shap_start
            _LOG.info(
                "SHAP completed | model=%s | method=%s | samples=%s | background=%d | %.2fs",
                model_name,
                shap_importance_method or "n/a",
                config.shap_max_samples if config.shap_max_samples is not None else "all",
                config.shap_background_samples,
                shap_elapsed,
            )

    y_train_true = _ensure_2d(_unscale_targets(prepared.target_scaler, splits.y_train))
    y_val_true = _ensure_2d(_unscale_targets(prepared.target_scaler, splits.y_val))
    y_test_true = _ensure_2d(_unscale_targets(prepared.target_scaler, splits.y_test))

    y_train_pred = _ensure_2d(_unscale_targets(prepared.target_scaler, pred_train))
    y_val_pred = _ensure_2d(_unscale_targets(prepared.target_scaler, pred_val))
    y_test_pred = _ensure_2d(_unscale_targets(prepared.target_scaler, pred_test))

    aggregate_metrics: Dict[str, Dict[str, float]] = {}
    per_gene_metrics: Dict[str, List[Dict[str, float]]] = {}
    split_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    for split_name, cells, truth, pred in (
        ("train", splits.cell_ids_train, y_train_true, y_train_pred),
        ("val", splits.cell_ids_val, y_val_true, y_val_pred),
        ("test", splits.cell_ids_test, y_test_true, y_test_pred),
    ):
        agg, per_gene = _compute_multi_metrics(truth, pred, gene_names)
        aggregate_metrics[split_name] = agg
        per_gene_metrics[split_name] = per_gene
        split_predictions[split_name] = {
            "cell_ids": np.asarray(cells),
            "y_true": truth,
            "y_pred": pred,
        }

    _LOG.info(
        "Final metrics | model=%s | mean_train_R2=%.4f | mean_val_R2=%.4f | mean_test_R2=%.4f",
        model_name,
        aggregate_metrics["train"].get("r2", float("nan")),
        aggregate_metrics["val"].get("r2", float("nan")),
        aggregate_metrics["test"].get("r2", float("nan")),
    )

    return CellwiseModelResult(
        model_name=model_name,
        gene_names=gene_names,
        gene_infos=list(getattr(dataset, "genes", [])),
        cv_metrics=cv_metrics,
        aggregate_metrics=aggregate_metrics,
        per_gene_metrics=per_gene_metrics,
        split_predictions=split_predictions,
        fitted_model=fitted_model,
        history=history,
        feature_importances=feature_importances,
        feature_names=getattr(dataset, "feature_names", None),
        feature_importance_method=feature_importance_method,
        shap_importances_mean=shap_importances_mean,
        shap_importance_method=shap_importance_method,
        feature_block_slices=getattr(dataset, "feature_block_slices", None),
        feature_block_indices=getattr(dataset, "feature_block_indices", None),
        feature_scaler=prepared.feature_scaler,
        target_scaler=prepared.target_scaler,
        reshape=model.reshape if isinstance(model, TorchModelBundle) else None,
    )


def _fit_torch_model(
    bundle: TorchModelBundle,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
    *,
    target_scaler: Optional[StandardScaler | MinMaxScaler] = None,
    capture_history: bool = False,
) -> Tuple[nn.Module, np.ndarray, Optional[List[Dict[str, float]]]]:
    device = _select_device(config.device_preference)
    model = bundle.model.to(device)
    model = _wrap_model_for_multi_gpu(model, device)

    def _sanitize_array(name: str, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if not np.isfinite(arr).all():
            _LOG.warning("Non-finite values detected in %s; replacing with 0.", name)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    X_train = _sanitize_array("X_train", X_train)
    X_val = _sanitize_array("X_val", X_val)
    y_train = _sanitize_array("y_train", y_train)
    y_val = _sanitize_array("y_val", y_val)

    y_train_arr = np.asarray(y_train)
    target_dim = y_train_arr.shape[1] if y_train_arr.ndim > 1 else 1

    train_ds = _make_dataset(bundle.reshape, X_train, y_train)
    val_ds = _make_dataset(bundle.reshape, X_val, y_val)
    batch_size = _effective_batch_size(config, target_dim)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    track_history = capture_history and config.track_history
    history: List[Dict[str, float]] = []
    train_eval_loader: Optional[DataLoader] = None
    if track_history:
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    use_amp = device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)

    best_state = None
    best_val = float("inf")
    patience = config.early_stopping_patience
    epochs_no_improve = 0
    process: Optional[Any] = None
    if track_history and psutil is not None:
        process = psutil.Process()
        try:
            process.cpu_percent(interval=None)
        except Exception:  # pragma: no cover - psutil quirks
            process = None

    def _collect_predictions(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        preds: List[np.ndarray] = []
        truths: List[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                with _amp_autocast(device.type, use_amp):
                    outputs = model(batch_x)
                preds.append(outputs.detach().cpu().numpy())
                truths.append(batch_y.detach().cpu().numpy())
        if not preds:
            return np.empty((0, target_dim), dtype=np.float32), np.empty((0, target_dim), dtype=np.float32)
        pred_arr = np.concatenate(preds, axis=0)
        truth_arr = np.concatenate(truths, axis=0)
        if pred_arr.ndim == 1:
            pred_arr = pred_arr.reshape(-1, 1)
        if truth_arr.ndim == 1:
            truth_arr = truth_arr.reshape(-1, 1)
        return pred_arr, truth_arr

    def _compute_metric_summary(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray) -> Dict[str, float]:
        if y_true_scaled.size == 0 or y_pred_scaled.size == 0:
            return {}
        y_true_unscaled = _unscale_targets(target_scaler, y_true_scaled)
        y_pred_unscaled = _unscale_targets(target_scaler, y_pred_scaled)
        y_true_arr = _ensure_2d(np.asarray(y_true_unscaled))
        y_pred_arr = _ensure_2d(np.asarray(y_pred_unscaled))
        if y_true_arr.shape[1] == 1:
            return regression_metrics(y_true_arr.ravel(), y_pred_arr.ravel())

        metrics_per_target = [
            regression_metrics(y_true_arr[:, idx], y_pred_arr[:, idx])
            for idx in range(y_true_arr.shape[1])
        ]
        if not metrics_per_target:
            return {}
        keys = metrics_per_target[0].keys()
        summary: Dict[str, float] = {}
        for key in keys:
            values = [entry.get(key, float("nan")) for entry in metrics_per_target]
            summary[key] = float(np.nanmean(values))
        return summary

    _LOG.info(
        "Starting training | model=%s | device=%s | epochs=%d | batch_size=%d",
        type(model).__name__,
        device.type,
        config.epochs,
        batch_size,
    )
    _log_gpu_memory_snapshot("Before training start")

    for epoch in range(config.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        train_loss_accum = 0.0
        train_samples = 0
        nan_detected = False
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _amp_autocast(device.type, use_amp):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            if not torch.isfinite(loss):
                _LOG.warning(
                    "Non-finite training loss detected at epoch %d; stopping early.",
                    epoch + 1,
                )
                nan_detected = True
                break
            scaler.scale(loss).backward()
            if config.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_size_curr = batch_x.size(0)
            train_loss_accum += float(loss.item()) * batch_size_curr
            train_samples += batch_size_curr

        if nan_detected:
            break

        model.eval()
        running = []
        val_preds_epoch: List[np.ndarray] = [] if track_history else []
        val_true_epoch: List[np.ndarray] = [] if track_history else []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                with _amp_autocast(device.type, use_amp):
                    preds = model(val_x)
                    val_loss = criterion(preds, val_y)
                running.append(val_loss.item())
                if track_history:
                    val_preds_epoch.append(preds.detach().cpu().numpy())
                    val_true_epoch.append(val_y.detach().cpu().numpy())
        mean_val = float(np.mean(running)) if running else best_val
        if not np.isfinite(mean_val):
            _LOG.warning(
                "Non-finite validation loss detected at epoch %d; stopping early.",
                epoch + 1,
            )
            break
        should_stop = False
        if mean_val < best_val - 1e-6:
            best_val = mean_val
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                should_stop = True

        gpu_util_pct = None
        if device.type == "cuda":
            gpu_util_pct = _get_gpu_utilization_pct()

        if track_history:
            train_loss_mean = train_loss_accum / max(train_samples, 1)
            train_pred_scaled, train_true_scaled = _collect_predictions(train_eval_loader)  # type: ignore[arg-type]
            if val_preds_epoch:
                val_pred_scaled = np.concatenate(val_preds_epoch, axis=0)
                val_true_scaled = np.concatenate(val_true_epoch, axis=0)
            else:
                val_pred_scaled, val_true_scaled = _collect_predictions(val_loader)

            train_metrics = _compute_metric_summary(train_true_scaled, train_pred_scaled)
            val_metrics = _compute_metric_summary(val_true_scaled, val_pred_scaled)

            entry: Dict[str, float] = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss_mean),
                "val_loss": float(mean_val),
                "effective_batch_size": float(batch_size),
            }
            if process is not None:
                try:
                    entry["cpu_percent"] = float(process.cpu_percent(interval=None))
                except Exception:
                    pass
                try:
                    if "gpu_util_pct" in locals() and gpu_util_pct is not None:
                        entry["rss_gib"] = float(torch.cuda.memory_allocated()) / (1024 ** 3)
                except Exception:
                    pass
                try:
                    entry["thread_count"] = float(process.num_threads())
                except Exception:
                    pass
            if device.type == "cuda":
                entry["gpu_alloc_mb"] = float(torch.cuda.memory_allocated() / (1024 ** 2))
                entry["gpu_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024 ** 2))
                entry["gpu_peak_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                if gpu_util_pct is not None:
                    entry["gpu_util_pct"] = float(gpu_util_pct)
            for metric_name in config.history_metrics:
                key = metric_name.lower()
                if key == "loss":
                    continue
                train_value = train_metrics.get(key)
                val_value = val_metrics.get(key)
                if train_value is not None:
                    entry[f"train_{key}"] = float(train_value)
                if val_value is not None:
                    entry[f"val_{key}"] = float(val_value)
            history.append(entry)

        gpu_peak_mb = None
        if device.type == "cuda":
            gpu_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

        if config.track_history and history:
            recent = history[-1]
            log_msg = (
                "Epoch %d/%d | model=%s | batch_size=%d | train_loss=%.6f | val_loss=%.6f"
                % (
                    epoch + 1,
                    config.epochs,
                    type(model).__name__,
                    batch_size,
                    recent.get("train_loss", float("nan")),
                    recent.get("val_loss", float("nan")),
                )
            )
            for metric_name in config.history_metrics:
                key = metric_name.lower()
                if key == "loss":
                    continue
                train_key = f"train_{key}"
                val_key = f"val_{key}"
                if train_key in recent:
                    log_msg += f" | {train_key}={recent[train_key]:.4f}"
                if val_key in recent:
                    log_msg += f" | {val_key}={recent[val_key]:.4f}"
            if "cpu_percent" in recent:
                log_msg += f" | cpu_pct={recent['cpu_percent']:.1f}"
            if "rss_gib" in recent:
                log_msg += f" | rss_gib={recent['rss_gib']:.2f}"
            if "thread_count" in recent:
                log_msg += f" | threads={recent['thread_count']:.0f}"
            if "gpu_alloc_mb" in recent:
                log_msg += f" | gpu_alloc_mb={recent['gpu_alloc_mb']:.0f}"
            if "gpu_reserved_mb" in recent:
                log_msg += f" | gpu_reserved_mb={recent['gpu_reserved_mb']:.0f}"
            if gpu_peak_mb is not None:
                log_msg += f" | gpu_peak_alloc_mb={gpu_peak_mb:.0f}"
            if gpu_util_pct is not None:
                log_msg += f" | gpu_util_pct={gpu_util_pct:.0f}"
            _LOG.info(log_msg)
        else:
            log_msg = (
                "Epoch %d/%d | model=%s | batch_size=%d | train_loss=%.6f | val_loss=%.6f"
                % (
                    epoch + 1,
                    config.epochs,
                    type(model).__name__,
                    batch_size,
                    train_loss_accum / max(train_samples, 1),
                    mean_val,
                )
            )
            if gpu_peak_mb is not None:
                log_msg += f" | gpu_peak_alloc_mb={gpu_peak_mb:.0f}"
            if gpu_util_pct is not None:
                log_msg += f" | gpu_util_pct={gpu_util_pct:.0f}"
            _LOG.info(log_msg)

        if should_stop:
            break

    _log_gpu_memory_snapshot("After training complete")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds = _predict_torch(model, bundle.reshape, config, X_val, output_dim=target_dim)
    return model.cpu(), preds, history if history else None


def _predict_torch(
    model: nn.Module,
    reshape: str,
    config: TrainingConfig,
    X: np.ndarray,
    output_dim: int = 1,
) -> np.ndarray:
    device = _select_device(config.device_preference)
    model = model.to(device)
    if output_dim > 1:
        placeholder = np.zeros((X.shape[0], output_dim), dtype=np.float32)
    else:
        placeholder = np.zeros(X.shape[0], dtype=np.float32)
    ds = _make_dataset(reshape, X, placeholder)
    batch_size = _effective_batch_size(config, output_dim)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    preds: List[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            with _amp_autocast(device.type, use_amp):
                outputs = model(batch_x)
            arr = outputs.detach().cpu().numpy()
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            preds.append(arr)
    result = np.concatenate(preds, axis=0)
    if result.ndim == 2 and result.shape[1] == 1:
        return result.ravel()
    return result


def _make_dataset(reshape: str, X: np.ndarray, y: np.ndarray) -> TensorDataset:
    if reshape == "sequence":
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(-1)
    y_array = np.asarray(y, dtype=np.float32)
    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)
    y_tensor = torch.tensor(y_array, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def _select_device(device_preference: str) -> torch.device:
    pref = (device_preference or "cuda").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            _LOG.info("Auto-selected CUDA device")
            device = torch.device("cuda")
            _log_gpu_memory_snapshot("CUDA device selected")
            return device
        _LOG.warning("CUDA not available; auto device falling back to CPU")
        return torch.device("cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            _log_gpu_memory_snapshot("CUDA device selected (explicit)")
            return device
        _LOG.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device("cpu")


def _effective_batch_size(config: TrainingConfig, target_dim: int) -> int:
    """
    Compute effective batch size, scaling down for multi-output models.
    
    The base batch size is taken from ``config.batch_size``. For multi-output
    models, the batch size is reduced so that the product
    ``target_dim * effective_batch_size`` does not exceed
    ``config.effective_batch_cap``. This helps manage memory usage across
    multiple targets.

    For single-output (``target_dim == 1``), this function returns the base
    batch size without scaling. ``target_dim`` values less than 1 are invalid
    and raise ``ValueError``.

    Args:
        config: Training configuration providing ``batch_size`` and
            ``effective_batch_cap`` attributes.
        target_dim: Number of output targets (e.g., genes) predicted by the
            model.

    Returns:
        Effective batch size to use for training.
    """
    if target_dim < 1:
        raise ValueError(f"target_dim must be >= 1, got {target_dim}")
    base = max(1, int(config.batch_size))
    if target_dim == 1:
        # Single-output: no scaling needed
        return base
    # Multi-output: scale batch size down to respect effective_batch_cap
    limit = max(8, int(config.effective_batch_cap) // target_dim)
    return max(8, min(base, limit))


def _unscale_targets(scaler: Optional[StandardScaler | MinMaxScaler], values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if scaler is None:
        return arr
    if arr.ndim == 1:
        inv = scaler.inverse_transform(arr.reshape(-1, 1))
        return inv.ravel()
    return scaler.inverse_transform(arr)


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _apply_knn_smoothing(
    X: np.ndarray,
    Y: np.ndarray,
    cell_ids: np.ndarray,
    *,
    group_size: int,
    n_components: int,
    random_state: int,
    split_label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-NN smoothing: average each cell with its k-1 nearest neighbors (dataset size unchanged)."""
    if X.size == 0:
        return X, Y, cell_ids

    n_cells = X.shape[0]
    if group_size <= 1 or n_cells <= 1:
        return X, Y, cell_ids

    components = max(1, min(n_components, X.shape[1], n_cells))
    if components < 1:
        return X, Y, cell_ids

    start_time = time.perf_counter()
    _log_resource_snapshot(f"smoothing:{split_label}:start")

    X_for_pca = X
    if X.shape[0] > 1 and X.shape[1] > 0:
        scaler = StandardScaler(with_mean=False)
        try:
            X_for_pca = scaler.fit_transform(X)
        except Exception:  # pragma: no cover - fallback to raw values
            X_for_pca = X
    try:
        pca = PCA(n_components=components, random_state=random_state)
        embedding = pca.fit_transform(X_for_pca)
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("PCA failed for %s split (%s); skipping smoothing", split_label, exc)
        return X, Y, cell_ids

    k_neighbors = min(group_size - 1, n_cells - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    nn.fit(embedding)
    _, neighbor_indices = nn.kneighbors(embedding)

    # Vectorized neighbor averaging for speed on large datasets
    neighbor_set = neighbor_indices[:, : k_neighbors + 1]
    X_smoothed = np.asarray(X[neighbor_set], dtype=np.float32).mean(axis=1)
    if Y.ndim == 1:
        Y_smoothed = np.asarray(Y[neighbor_set], dtype=np.float32).mean(axis=1)
    else:
        Y_smoothed = np.asarray(Y[neighbor_set], dtype=np.float32).mean(axis=1)

    elapsed = time.perf_counter() - start_time
    _LOG.info(
        "Smoothing applied to %s split: %d cells (k=%d | components=%d | %.2fs)",
        split_label,
        n_cells,
        k_neighbors,
        components,
        elapsed,
    )
    _log_resource_snapshot(f"smoothing:{split_label}:end")

    return X_smoothed, Y_smoothed, cell_ids


def _apply_pseudobulk(
    X: np.ndarray,
    Y: np.ndarray,
    cell_ids: np.ndarray,
    *,
    group_labels: np.ndarray,
    group_size: int,
    n_components: int,
    random_state: int,
    split_label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.size == 0:
        return X, Y, cell_ids, group_labels

    n_cells = X.shape[0]
    if group_size <= 1 or n_cells <= 1:
        return X, Y, cell_ids, group_labels

    components = max(1, min(n_components, X.shape[1], n_cells))
    if components < 1:
        return X, Y, cell_ids, group_labels

    start_time = time.perf_counter()
    _log_resource_snapshot(f"pseudobulk:{split_label}:start")

    rng = np.random.default_rng(random_state)

    X_for_pca = X
    if X.shape[0] > 1 and X.shape[1] > 0:
        scaler = StandardScaler(with_mean=False)
        try:
            X_for_pca = scaler.fit_transform(X)
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("PCA preprocessing failed for %s split (%s); using raw features", split_label, exc)
            X_for_pca = X
    try:
        pca = PCA(n_components=components, random_state=random_state)
        embedding = pca.fit_transform(X_for_pca)
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("PCA failed for %s split (%s); skipping pseudobulk", split_label, exc)
        return X, Y, cell_ids, group_labels

    neighbor_pool = min(n_cells, max(group_size * 5, group_size))
    nn = NearestNeighbors(n_neighbors=neighbor_pool, metric="euclidean")
    nn.fit(embedding)

    assigned = np.zeros(n_cells, dtype=bool)
    order = rng.permutation(n_cells)
    groups: List[List[int]] = []
    bulk_group_labels: List[str] = []
    group_labels_arr = np.asarray(group_labels)
    group_labels_str = group_labels_arr.astype(str)

    for seed in order:
        if assigned[seed]:
            continue
        seed_group = group_labels_str[int(seed)]
        group: List[int] = [int(seed)]
        assigned[int(seed)] = True
        neighbors = nn.kneighbors(embedding[seed : seed + 1], return_distance=False)
        for neighbor in neighbors[0]:
            neighbor_idx = int(neighbor)
            if neighbor_idx == seed or assigned[neighbor_idx]:
                continue
            if group_labels_str[neighbor_idx] != seed_group:
                continue
            group.append(neighbor_idx)
            assigned[neighbor_idx] = True
            if len(group) >= group_size:
                break
        if len(group) < group_size:
            remaining = np.where((~assigned) & (group_labels_str == seed_group))[0]
            if remaining.size:
                extra_count = min(group_size - len(group), remaining.size)
                extra_indices = rng.choice(remaining, size=extra_count, replace=False)
                for idx in extra_indices:
                    if assigned[int(idx)]:
                        continue
                    group.append(int(idx))
                    assigned[int(idx)] = True
        groups.append(group)
        bulk_group_labels.append(seed_group)

    leftover = np.where(~assigned)[0]
    if leftover.size:
        for idx in leftover:
            idx_group = group_labels_str[int(idx)]
            same_group_targets = [g_idx for g_idx, label in enumerate(bulk_group_labels) if label == idx_group]
            if same_group_targets:
                target_group = int(rng.choice(same_group_targets))
            else:
                target_group = int(rng.integers(low=0, high=len(groups)))
                if target_group >= len(bulk_group_labels):
                    bulk_group_labels.append(idx_group)
                else:
                    bulk_group_labels[target_group] = idx_group
            groups[target_group].append(int(idx))
            assigned[int(idx)] = True

    X_bulk: List[np.ndarray] = []
    y_is_vector = Y.ndim == 1
    Y_bulk: List[np.ndarray] = []
    bulk_ids: List[str] = []

    for grp_idx, grp in enumerate(groups):
        indices = np.asarray(grp, dtype=int)
        X_bulk.append(np.asarray(X[indices], dtype=np.float64).mean(axis=0))
        if y_is_vector:
            Y_bulk.append(np.asarray(Y[indices], dtype=np.float64).mean(axis=0, keepdims=True))
        else:
            Y_bulk.append(np.asarray(Y[indices], dtype=np.float64).mean(axis=0))
        bulk_ids.append(f"{split_label}_bulk_{grp_idx:05d}")

    X_out = np.vstack(X_bulk).astype(np.float32)
    if y_is_vector:
        Y_out = np.vstack(Y_bulk).ravel().astype(np.float32)
    else:
        Y_out = np.vstack(Y_bulk).astype(np.float32)
    cell_out = np.asarray(bulk_ids, dtype=str)
    groups_out = np.asarray(bulk_group_labels if bulk_group_labels else group_labels_arr, dtype=str)

    elapsed = time.perf_counter() - start_time
    _LOG.info(
        "Pseudobulked %s split: %d cells -> %d groups (target size=%d | components=%d | %.2fs)",
        split_label,
        n_cells,
        len(groups),
        group_size,
        components,
        elapsed,
    )
    _log_resource_snapshot(f"pseudobulk:{split_label}:end")

    return X_out, Y_out, cell_out, groups_out


def _compute_multi_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_names: Sequence[str],
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    y_true_2d = _ensure_2d(y_true)
    y_pred_2d = _ensure_2d(y_pred)

    per_gene: List[Dict[str, float]] = []
    for idx, gene in enumerate(gene_names):
        metrics = regression_metrics(y_true_2d[:, idx], y_pred_2d[:, idx])
        metrics_with_gene = dict(metrics)
        metrics_with_gene["gene"] = gene
        per_gene.append(metrics_with_gene)

    metric_keys = ["pearson", "r2", "spearman", "rmse", "mse", "mae"]
    aggregate = {
        key: float(np.nanmean([entry.get(key, float("nan")) for entry in per_gene]))
        for key in metric_keys
    }
    return aggregate, per_gene


def _stack_predictions(
    gene_name: str,
    model_name: str,
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.recarray:
    rows: List[tuple] = []
    for split, (cell_ids, y_true, y_pred) in predictions.items():
        for cid, truth, pred in zip(cell_ids, y_true, y_pred, strict=False):
            rows.append((gene_name, model_name, split, cid, float(truth), float(pred)))
    dtype = np.dtype(
        [
            ("gene", "U64"),
            ("model", "U32"),
            ("split", "U16"),
            ("cell_id", "U64"),
            ("y_true", "f8"),
            ("y_pred", "f8"),
        ]
    )
    return np.array(rows, dtype=dtype).view(np.recarray)

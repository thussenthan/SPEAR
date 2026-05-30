import copy
import hashlib
import json
import math
import random
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import contextlib

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    train_test_split,
)
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
    cache_key_lock,
    save_prepared_data,
    load_prepared_data,
    save_prepared_cellwise_data,
    load_prepared_cellwise_data,
)
from .data_types import PreparedData, PreparedCellwiseData, SplitData, CellwiseSplitData
from .wandb_utils import wandb_log_metrics

# Suppress specific warnings that are informational only
warnings.filterwarnings(
    "ignore", message=".*CuDNN issue.*nvrtc.so.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=".*Ill-conditioned matrix.*", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*sklearn.utils.parallel.Parallel.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Implicitly cleaning up.*TemporaryDirectory.*",
    category=ResourceWarning,
)


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
_FAST_CLASSICAL_MODELS = {
    "svr",
    "lasso",
    "elastic_net",
    "hist_gradient_boosting",
    "catboost",
}
FAST_MODE_MIN_PSEUDOBULK_GROUP_SIZE = 8


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


if (
    hasattr(torch, "amp")
    and hasattr(torch.amp, "GradScaler")
    and hasattr(torch.amp, "autocast")
):
    _AMP_GRAD_SCALER = torch.amp.GradScaler
    _AMP_AUTOCAST = torch.amp.autocast
else:  # pragma: no cover - AMP unavailable
    _AMP_GRAD_SCALER = None
    _AMP_AUTOCAST = None


def _make_grad_scaler(use_amp: bool):
    if not use_amp or _AMP_GRAD_SCALER is None:
        return _NoopGradScaler()
    return _AMP_GRAD_SCALER(enabled=True)


def _sanitize_numeric_array(
    name: str, arr: np.ndarray, *, fill_value: float = 0.0
) -> np.ndarray:
    array = np.asarray(arr)
    if not np.isfinite(array).all():
        _LOG.warning(
            "Non-finite values detected in %s; replacing with %.1f.", name, fill_value
        )
        array = np.nan_to_num(
            array, nan=fill_value, posinf=fill_value, neginf=fill_value
        )
    return array


def _snapshot_model_state(model: nn.Module) -> Dict[str, object]:
    snapshot: Dict[str, object] = {}
    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            snapshot[key] = value.detach().cpu().clone()
        else:
            snapshot[key] = copy.deepcopy(value)
    return snapshot


def _fit_single_output_estimator(
    estimator: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_name: str,
    gene_name: str,
    fallback_reasons: Optional[List[str]] = None,
) -> object:
    X_arr = np.asarray(X_train)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y_train).ravel()

    feature_has_variation = X_arr.shape[1] > 0 and bool(
        np.any(np.ptp(X_arr, axis=0) > 0)
    )
    target_is_constant = y_arr.size == 0 or np.allclose(
        y_arr, y_arr[0], equal_nan=False
    )

    if not feature_has_variation or target_is_constant:
        reasons: List[str] = []
        if not feature_has_variation:
            reasons.append("all features constant")
        if target_is_constant:
            reasons.append("target constant")
        if fallback_reasons is not None:
            fallback_reasons.append(
                f"{model_name}:{gene_name}:DummyRegressor fallback ({', '.join(reasons)})"
            )
        _LOG.warning(
            "Using DummyRegressor fallback | model=%s | gene=%s | reason=%s",
            model_name,
            gene_name,
            ", ".join(reasons),
        )
        fallback = DummyRegressor(strategy="mean")
        fallback.fit(X_arr, y_arr)
        return fallback

    try:
        estimator.fit(X_arr, y_arr)
        return estimator
    except Exception as exc:
        error_text = str(exc).lower()
        if (
            "all features are either constant or ignored" not in error_text
            and "all train targets are equal" not in error_text
        ):
            raise
        if fallback_reasons is not None:
            fallback_reasons.append(
                f"{model_name}:{gene_name}:DummyRegressor fallback after fit failure ({exc})"
            )
        _LOG.warning(
            "Using DummyRegressor fallback after %s fit failure | gene=%s | error=%s",
            model_name,
            gene_name,
            exc,
        )
        fallback = DummyRegressor(strategy="mean")
        fallback.fit(X_arr, y_arr)
        return fallback


def _has_feature_variation(X: np.ndarray) -> bool:
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.size == 0 or X_arr.shape[1] == 0:
        return False
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return bool(np.any(np.ptp(X_arr, axis=0) > 0))


def _target_is_constant(y: np.ndarray) -> bool:
    y_arr = np.asarray(y).ravel()
    y_arr = y_arr[np.isfinite(y_arr)]
    if y_arr.size < 2:
        return True
    return bool(np.allclose(y_arr, y_arr[0], atol=1e-8, rtol=1e-5))


def _torch_dummy_reason_single_output(
    X_train: np.ndarray, y_train: np.ndarray
) -> Optional[str]:
    reasons: List[str] = []
    if not _has_feature_variation(X_train):
        reasons.append("all features constant")
    if _target_is_constant(y_train):
        reasons.append("target constant")
    if np.asarray(X_train).shape[0] < 2:
        reasons.append("insufficient train samples")
    if reasons:
        return ", ".join(reasons)
    return None


def _fit_dummy_single_output(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> Tuple[DummyRegressor, np.ndarray]:
    fallback = DummyRegressor(strategy="mean")
    fallback.fit(np.asarray(X_train), np.asarray(y_train).ravel())
    preds = fallback.predict(np.asarray(X_eval))
    return fallback, np.asarray(preds).ravel()


def _amp_autocast(device_type: str, use_amp: bool):
    if not use_amp or _AMP_AUTOCAST is None:
        return contextlib.nullcontext()
    return _AMP_AUTOCAST(device_type=device_type, enabled=True)


def _reshape_tensor_for_model(tens: torch.Tensor, reshape: str | None) -> torch.Tensor:
    """Utility to reshape tensor for model input, shared between training and prediction."""
    if reshape == "sequence":
        if tens.dim() == 3:
            if tens.size(-1) == 1:
                return tens.transpose(1, 2)
            return tens
        return tens.reshape(tens.shape[0], 1, -1)
    return tens


def _sequence_distance_kwargs(
    dataset,
    config: TrainingConfig,
    reshape: str | None,
) -> Dict[str, Optional[np.ndarray]]:
    if (
        reshape != "sequence"
        or getattr(config, "per_gene_feature_basis", "bin") != "peak"
        or getattr(config, "per_gene_peak_distance_encoding", "none") == "none"
    ):
        return {"sequence_offsets": None, "sequence_offset_features": None}

    offsets: list[float] = []
    for name in getattr(dataset, "feature_names", []) or []:
        if "|offset_" not in name:
            offsets.append(0.0)
            continue
        suffix = name.split("|offset_", 1)[1]
        if suffix.endswith("bp"):
            suffix = suffix[:-2]
        try:
            offsets.append(float(int(suffix)))
        except Exception:
            offsets.append(0.0)

    if not offsets:
        return {"sequence_offsets": None, "sequence_offset_features": None}

    window_bp = float(max(1, int(getattr(config, "window_bp", 1))))
    offsets_arr = np.asarray(offsets, dtype=np.float32)
    sequence_offsets = (offsets_arr / window_bp).astype(np.float32)
    enc = str(
        getattr(config, "per_gene_peak_distance_encoding", "none") or "none"
    ).lower()
    if enc == "signed_linear":
        sequence_offset_features = sequence_offsets.reshape(-1, 1)
    elif enc == "rbf":
        k = int(getattr(config, "per_gene_peak_distance_rbf_bases", 16))
        gamma = float(getattr(config, "per_gene_peak_distance_rbf_gamma", 4.0))
        signed_log = np.sign(offsets_arr) * (
            np.log1p(np.abs(offsets_arr)) / np.log1p(window_bp)
        )
        centers = np.linspace(-1.0, 1.0, num=k, dtype=np.float32)
        diffs = signed_log.astype(np.float32).reshape(-1, 1) - centers.reshape(1, -1)
        rbf = np.exp(-gamma * np.square(diffs)).astype(np.float32)
        sequence_offset_features = np.concatenate(
            [sequence_offsets.reshape(-1, 1), rbf],
            axis=1,
        ).astype(np.float32)
    else:
        sequence_offset_features = None

    return {
        "sequence_offsets": sequence_offsets,
        "sequence_offset_features": sequence_offset_features,
    }


_LOG = get_logger(__name__)


def _wrap_model_for_multi_gpu(model: nn.Module, device: torch.device) -> nn.Module:
    """Wrap model in DataParallel if multiple GPUs are available and device is CUDA."""
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        _LOG.info(
            "Wrapping model in DataParallel for %d GPUs", torch.cuda.device_count()
        )
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
        rss_gib = rss_bytes / (1024**3)
        _RESOURCE_TRACKER["peak_rss_gib"] = max(
            _RESOURCE_TRACKER["peak_rss_gib"], rss_gib
        )
    except Exception:  # pragma: no cover - defensive fallback
        rss_gib = float("nan")
    try:
        if not _CPU_PRIMED:
            # Prime the CPU percent sampler and use the priming sample as the first value.
            cpu_pct = process.cpu_percent(interval=0.1)
            _CPU_PRIMED = True
        else:
            cpu_pct = process.cpu_percent(interval=None)
        _RESOURCE_TRACKER["peak_cpu_pct"] = max(
            _RESOURCE_TRACKER["peak_cpu_pct"], cpu_pct
        )
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

        allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.memory_reserved() / (1024**2)

        # Peak memory since last reset
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)

        # Available on device
        device_count = torch.cuda.device_count()
        total_device_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        free_device_mb = total_device_mb - reserved_mb

        # Update tracker
        _RESOURCE_TRACKER["peak_gpu_allocated_mb"] = max(
            _RESOURCE_TRACKER["peak_gpu_allocated_mb"], peak_allocated_mb
        )
        _RESOURCE_TRACKER["peak_gpu_reserved_mb"] = max(
            _RESOURCE_TRACKER["peak_gpu_reserved_mb"], reserved_mb
        )
        _RESOURCE_TRACKER["peak_gpu_free_mb"] = min(
            _RESOURCE_TRACKER["peak_gpu_free_mb"], free_device_mb
        )
        _RESOURCE_TRACKER["max_gpu_devices"] = max(
            _RESOURCE_TRACKER["max_gpu_devices"], device_count
        )

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


def _config_cache_key(
    config: TrainingConfig,
    scope: str,
    *,
    dataset_key: Optional[str] = None,
    use_global_knn: bool = False,
) -> str:
    payload = {
        "scope": scope,
        "prep_version": 4,
        "use_global_knn": use_global_knn,
        # Feature geometry — changing these changes the feature matrix shape entirely
        "window_bp": config.window_bp,
        "bin_size_bp": config.bin_size_bp,
        "per_gene_feature_basis": getattr(config, "per_gene_feature_basis", "bin"),
        # Input layers — changing these changes what signal is used
        "atac_layer": config.atac_layer or "none",
        "rna_expression_layer": config.rna_expression_layer or "none",
        # Cell/gene filtering thresholds
        "min_cells_per_gene": config.min_cells_per_gene,
        "min_expression": config.min_expression,
        # Target normalization — these three interact to determine what Y values are cached
        "log1p_transform": bool(config.log1p_transform),
        "target_scaler": config.target_scaler or "none",
        "force_target_scaling": bool(getattr(config, "force_target_scaling", False)),
        # Data splitting and smoothing
        "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
        "test_fraction": config.test_fraction,
        "random_state": config.random_state,
        "group_key": config.group_key,
        "enable_smoothing": config.enable_smoothing,
        "smoothing_k": config.smoothing_k,
        "smoothing_pca_components": config.smoothing_pca_components,
        "smoothing_target": getattr(config, "smoothing_target", "all_splits"),
        "smoothing_y": bool(getattr(config, "smoothing_y", True)),
        "global_atac_components": int(
            getattr(config, "global_atac_components", 0) or 0
        ),
        "pseudobulk_group_size": config.pseudobulk_group_size,
        "pseudobulk_pca_components": config.pseudobulk_pca_components,
        "scaler": config.scaler or "none",
        "per_gene_cell_filter_mode": getattr(
            config, "per_gene_cell_filter_mode", "auto"
        ),
    }
    if dataset_key:
        payload["dataset_key"] = dataset_key
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
    data_counts: Dict[str, object] | None = None
    used_fallback: bool = False
    fallback_reasons: List[str] = field(default_factory=list)


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
    feature_importance_mean_signed: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    feature_importance_method: Optional[str] = None
    shap_importance_mean: Optional[np.ndarray] = None
    shap_value_mean_signed: Optional[np.ndarray] = None
    shap_importance_method: Optional[str] = None
    feature_block_slices: Optional[List[Tuple[int, int]]] = None
    feature_block_indices: Optional[List[np.ndarray]] = None
    feature_scaler: Optional[StandardScaler | MinMaxScaler] = None
    target_scaler: Optional[StandardScaler | MinMaxScaler] = None
    reshape: Optional[str] = None
    used_fallback: bool = False
    fallback_reasons: List[str] = field(default_factory=list)


@dataclass
class GlobalSplitKNN:
    """Pre-computed train/val/test splits and kNN neighbor indices from the full filtered ATAC matrix.

    In per-gene mode the default smoothing computes PCA+kNN from each gene's 40-bin local
    window, which is a very weak neighbourhood signal.  Precomputing a single graph from
    the full (all-gene-window union) ATAC matrix before the per-gene loop gives every gene's
    smoothing step access to a biologically richer neighbourhood representation.
    """

    train_cell_ids: np.ndarray  # [n_train]
    val_cell_ids: np.ndarray  # [n_val]
    test_cell_ids: np.ndarray  # [n_test]
    train_neighbor_idx: (
        np.ndarray
    )  # [n_train, k] — indices into train split, first col = self
    val_neighbor_idx: np.ndarray  # [n_val, k]
    test_neighbor_idx: np.ndarray  # [n_test, k]
    group_labels_train: np.ndarray
    group_labels_val: np.ndarray
    group_labels_test: np.ndarray


def prepare_data(
    dataset,
    config: TrainingConfig,
    cache_dir: Optional[Path] = None,
    global_knn: Optional[GlobalSplitKNN] = None,
) -> PreparedData:
    dataset_key = None
    if hasattr(dataset, "gene") and getattr(dataset.gene, "gene_name", None):
        dataset_key = str(
            getattr(dataset.gene, "gene_id", "") or getattr(dataset.gene, "gene_name")
        )
    cache_key = _config_cache_key(
        config, "gene", dataset_key=dataset_key, use_global_knn=(global_knn is not None)
    )
    cache_dict = getattr(dataset, "prepared_cache", {})
    if cache_dict is None:
        cache_dict = {}
        setattr(dataset, "prepared_cache", cache_dict)

    # Try in-memory cache first
    if cache_key in cache_dict:
        return cache_dict[cache_key]  # type: ignore[return-value]

    # Try disk cache if directory provided
    if cache_dir is not None:
        with cache_key_lock(cache_dir, cache_key, "gene"):
            disk_cached = load_prepared_data(cache_dir, cache_key)
            if disk_cached is not None:
                # Restore to in-memory cache too
                cache_dict[cache_key] = disk_cached
                return disk_cached

            prepared_data = _prepare_data_uncached(
                dataset, config, global_knn=global_knn
            )
            try:
                save_prepared_data(prepared_data, cache_dir, cache_key)
            except Exception as exc:
                _LOG.warning("Failed to save prepared data to disk cache: %s", exc)
            cache_dict[cache_key] = prepared_data
            return prepared_data

    prepared_data = _prepare_data_uncached(dataset, config, global_knn=global_knn)
    cache_dict[cache_key] = prepared_data
    return prepared_data


def _prepare_data_uncached(
    dataset,
    config: TrainingConfig,
    *,
    global_knn: Optional[GlobalSplitKNN] = None,
) -> PreparedData:

    _seed_everything(config.random_state)

    _prep_start = time.perf_counter()
    _log_resource_snapshot("prepare_data:start")
    # Use genes[0].gene_name if available, else fallback to 'unknown'.
    if (
        hasattr(dataset, "genes")
        and dataset.genes
        and hasattr(dataset.genes[0], "gene_name")
    ):
        gene_name = dataset.genes[0].gene_name
    else:
        try:
            gene_name = dataset.gene.gene_name
        except Exception:
            gene_name = "unknown"
    dataset_metadata = dict(getattr(dataset, "metadata", {}) or {})
    _LOG.info(
        "Preparing dataset for gene %s | cells_raw=%d | cells_eligible=%d | cells_filtered_out=%d | filtering=%s | override=%s | features=%d",
        gene_name,
        int(dataset_metadata.get("cells_raw", dataset.num_cells())),
        int(dataset_metadata.get("cells_eligible", dataset.num_cells())),
        int(dataset_metadata.get("cells_filtered_out", 0)),
        "on" if dataset_metadata.get("cell_filtering_applied", False) else "off",
        str(dataset_metadata.get("cell_filtering_override_reason", "")) or "none",
        (
            dataset.num_features()
            if hasattr(dataset, "num_features")
            else int(dataset.X.shape[1])
        ),
    )

    X = dataset.X.astype(np.float32)
    y = dataset.y.astype(np.float32)
    cells = dataset.cell_ids
    groups = getattr(dataset, "group_labels", None)
    if groups is None:
        groups = np.asarray(cells)
    else:
        groups = np.asarray(groups)

    rng_state = config.random_state

    # --- Splitting ---
    # If a precomputed global split+kNN is available (per-gene mode with enable_smoothing),
    # use it directly.  This gives every gene the same cell assignment derived from the full
    # filtered ATAC matrix rather than each gene re-deriving its own split.  Fall back to
    # the standard per-gene local split when cell IDs don't align perfectly (e.g., if cell
    # filtering removed cells that are present in the global split).
    use_global_knn_split = False
    if global_knn is not None:
        _cell_id_to_local: Dict[str, int] = {cid: i for i, cid in enumerate(cells)}
        _train_local = np.array(
            [
                _cell_id_to_local[cid]
                for cid in global_knn.train_cell_ids
                if cid in _cell_id_to_local
            ],
            dtype=np.int64,
        )
        _val_local = np.array(
            [
                _cell_id_to_local[cid]
                for cid in global_knn.val_cell_ids
                if cid in _cell_id_to_local
            ],
            dtype=np.int64,
        )
        _test_local = np.array(
            [
                _cell_id_to_local[cid]
                for cid in global_knn.test_cell_ids
                if cid in _cell_id_to_local
            ],
            dtype=np.int64,
        )
        if (
            len(_train_local) == len(global_knn.train_cell_ids)
            and len(_val_local) == len(global_knn.val_cell_ids)
            and len(_test_local) == len(global_knn.test_cell_ids)
        ):
            X_train = X[_train_local]
            X_val = X[_val_local]
            X_test = X[_test_local]
            y_train = y[_train_local]
            y_val = y[_val_local]
            y_test = y[_test_local]
            cell_train = cells[_train_local]
            cell_val = cells[_val_local]
            cell_test = cells[_test_local]
            group_train = global_knn.group_labels_train
            group_val = global_knn.group_labels_val
            group_test = global_knn.group_labels_test
            split_strategy = "global_knn"
            use_global_knn_split = True
        else:
            _LOG.warning(
                "Global kNN cell IDs don't fully align with gene dataset "
                "(%d/%d train, %d/%d val, %d/%d test); falling back to local splits",
                len(_train_local),
                len(global_knn.train_cell_ids),
                len(_val_local),
                len(global_knn.val_cell_ids),
                len(_test_local),
                len(global_knn.test_cell_ids),
            )

    if not use_global_knn_split:
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
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=config.test_fraction, random_state=rng_state
            )
            try:
                train_val_idx, test_idx = next(splitter.split(X, y, groups))
            except ValueError:
                _LOG.warning(
                    "Falling back to random train/test split due to insufficient groups"
                )
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
            (
                X_temp,
                X_test,
                y_temp,
                y_test,
                cell_temp,
                cell_test,
                group_temp,
                group_test,
            ) = train_test_split(
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
            val_splitter = GroupShuffleSplit(
                n_splits=1, test_size=val_ratio, random_state=rng_state + 1
            )
            try:
                train_idx_rel, val_idx_rel = next(
                    val_splitter.split(X_temp, y_temp, group_temp)
                )
            except ValueError:
                _LOG.warning(
                    "Falling back to random train/val split due to insufficient groups"
                )
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
            (
                X_train,
                X_val,
                y_train,
                y_val,
                cell_train,
                cell_val,
                group_train,
                group_val,
            ) = train_test_split(
                X_temp,
                y_temp,
                cell_temp,
                group_temp,
                test_size=val_ratio,
                random_state=rng_state + 1,
            )
            group_train = np.asarray(group_train)
            group_val = np.asarray(group_val)

        split_strategy = "group" if use_group_split else "random"

    train_cells_raw = int(X_train.shape[0])
    val_cells_raw = int(X_val.shape[0])
    test_cells_raw = int(X_test.shape[0])

    smoothing_target = getattr(config, "smoothing_target", "all_splits")
    smooth_y = bool(getattr(config, "smoothing_y", True))
    if (
        config.enable_smoothing
        and config.smoothing_k > 1
        and smoothing_target != "none"
    ):
        smooth_train = smoothing_target in {"train_only", "all_splits"}
        smooth_eval = smoothing_target == "all_splits"
        if use_global_knn_split:
            if smooth_train:
                X_train, y_train, cell_train = _apply_knn_smoothing(
                    X_train,
                    y_train,
                    cell_train,
                    group_size=config.smoothing_k,
                    n_components=config.smoothing_pca_components,
                    random_state=config.random_state,
                    split_label="train",
                    precomputed_neighbor_idx=global_knn.train_neighbor_idx,
                    smooth_y=smooth_y,
                )
            if smooth_eval:
                X_val, y_val, cell_val = _apply_knn_smoothing(
                    X_val,
                    y_val,
                    cell_val,
                    group_size=config.smoothing_k,
                    n_components=config.smoothing_pca_components,
                    random_state=config.random_state + 1,
                    split_label="val",
                    precomputed_neighbor_idx=global_knn.val_neighbor_idx,
                    smooth_y=smooth_y,
                )
                X_test, y_test, cell_test = _apply_knn_smoothing(
                    X_test,
                    y_test,
                    cell_test,
                    group_size=config.smoothing_k,
                    n_components=config.smoothing_pca_components,
                    random_state=config.random_state + 2,
                    split_label="test",
                    precomputed_neighbor_idx=global_knn.test_neighbor_idx,
                    smooth_y=smooth_y,
                )
        else:
            if smooth_train:
                X_train, y_train, cell_train = _apply_knn_smoothing(
                    X_train,
                    y_train,
                    cell_train,
                    group_size=config.smoothing_k,
                    n_components=config.smoothing_pca_components,
                    random_state=config.random_state,
                    split_label="train",
                    smooth_y=smooth_y,
                )
            if smooth_eval:
                X_val, y_val, cell_val = _apply_knn_smoothing(
                    X_val,
                    y_val,
                    cell_val,
                    group_size=config.smoothing_k,
                    n_components=config.smoothing_pca_components,
                    random_state=config.random_state + 1,
                    split_label="val",
                    smooth_y=smooth_y,
                )
                X_test, y_test, cell_test = _apply_knn_smoothing(
                    X_test,
                    y_test,
                    cell_test,
                    group_size=config.smoothing_k,
                    n_components=config.smoothing_pca_components,
                    random_state=config.random_state + 2,
                    split_label="test",
                    smooth_y=smooth_y,
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
    train_cells_effective = int(X_train.shape[0])

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
        metadata={
            **dataset_metadata,
            "train_cells_raw": train_cells_raw,
            "val_cells_raw": val_cells_raw,
            "test_cells_raw": test_cells_raw,
            "train_cells_effective": train_cells_effective,
            "split_strategy": split_strategy,
        },
    )
    _LOG.info(
        "Prepared gene-wise splits | strategy=%s | train_raw=%d | val_raw=%d | test_raw=%d | train_effective=%d | val=%d | test=%d | features=%d | %.2fs",
        split_strategy,
        train_cells_raw,
        val_cells_raw,
        test_cells_raw,
        train_cells_effective,
        X_val.shape[0],
        X_test.shape[0],
        X_train.shape[1],
        time.perf_counter() - _prep_start,
    )
    _log_resource_snapshot("prepare_data:end")

    prepared_data = PreparedData(
        splits=splits,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )

    return prepared_data


def _cellwise_cache_key(dataset, config: TrainingConfig) -> str:
    cache_key = _config_cache_key(config, "cellwise")
    # Make cache dataset-specific to avoid collisions across datasets with same config.
    try:
        gene_names = [str(g.gene_name) for g in getattr(dataset, "genes", [])]
        if gene_names:
            gene_sig = hashlib.sha256("\n".join(gene_names).encode("utf-8")).hexdigest()
            sig = f"{dataset.X.shape}-{dataset.y.shape}-genes={gene_sig}"
        else:
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


def prepare_cellwise_data(
    dataset, config: TrainingConfig, cache_dir: Optional[Path] = None
) -> PreparedCellwiseData:
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
        with cache_key_lock(cache_dir, cache_key, "cellwise"):
            disk_cached = load_prepared_cellwise_data(cache_dir, cache_key)
            if disk_cached is not None:
                # Restore to in-memory cache too
                cache_dict[cache_key] = disk_cached
                return disk_cached

            prepared_cellwise = _prepare_cellwise_data_uncached(dataset, config)
            try:
                save_prepared_cellwise_data(prepared_cellwise, cache_dir, cache_key)
            except Exception as exc:
                _LOG.warning(
                    "Failed to save prepared cell-wise data to disk cache: %s", exc
                )
            cache_dict[cache_key] = prepared_cellwise
            return prepared_cellwise

    prepared_cellwise = _prepare_cellwise_data_uncached(dataset, config)
    cache_dict[cache_key] = prepared_cellwise
    return prepared_cellwise


def _prepare_cellwise_data_uncached(
    dataset, config: TrainingConfig
) -> PreparedCellwiseData:

    _LOG.info(
        "Preparing cell-wise data | cells_raw=%d | cells_eligible=%d | cells_filtered_out=%d | filtering=%s | override=%s | features=%d | targets=%d",
        int(getattr(dataset, "metadata", {}).get("cells_raw", dataset.X.shape[0])),
        int(getattr(dataset, "metadata", {}).get("cells_eligible", dataset.X.shape[0])),
        int(getattr(dataset, "metadata", {}).get("cells_filtered_out", 0)),
        (
            "on"
            if getattr(dataset, "metadata", {}).get("cell_filtering_applied", False)
            else "off"
        ),
        str(getattr(dataset, "metadata", {}).get("cell_filtering_override_reason", ""))
        or "none",
        dataset.X.shape[1],
        dataset.y.shape[1] if dataset.y.ndim > 1 else 1,
    )
    _log_resource_snapshot("prepare_cellwise_data:start")
    _cellwise_prep_start = time.perf_counter()

    X = dataset.X
    force_dense = getattr(config, "force_dense_features", True)
    needs_dense = (
        bool(config.enable_smoothing and config.smoothing_k > 1)
        or config.pseudobulk_group_size > 1
    )
    if sp.issparse(X) and not force_dense and needs_dense:
        _LOG.warning(
            "Sparse features with smoothing/pseudobulk require dense arrays; forcing densification."
        )
        force_dense = True
    if sp.issparse(X) and not force_dense and config.scaler == "minmax":
        _LOG.warning(
            "MinMax scaling does not support sparse inputs; forcing densification."
        )
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
            one_gib = 1024**3
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
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=config.test_fraction, random_state=rng_state
        )
        try:
            train_val_idx, test_idx = next(splitter.split(X, Y, groups))
        except ValueError:
            _LOG.warning(
                "Falling back to random train/test split for cellwise data due to insufficient groups"
            )
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
        X_temp, X_test, Y_temp, Y_test, cell_temp, cell_test, group_temp, group_test = (
            train_test_split(
                X,
                Y,
                cells,
                groups,
                test_size=config.test_fraction,
                random_state=rng_state,
            )
        )
        group_temp = np.asarray(group_temp)
        group_test = np.asarray(group_test)

    val_ratio = config.val_fraction / (config.train_fraction + config.val_fraction)

    if use_group_split:
        val_splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=rng_state + 1
        )
        try:
            train_idx_rel, val_idx_rel = next(
                val_splitter.split(X_temp, Y_temp, group_temp)
            )
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
        X_train, X_val, Y_train, Y_val, cell_train, cell_val, group_train, group_val = (
            train_test_split(
                X_temp,
                Y_temp,
                cell_temp,
                group_temp,
                test_size=val_ratio,
                random_state=rng_state + 1,
            )
        )
        group_train = np.asarray(group_train)
        group_val = np.asarray(group_val)

    smoothing_target = getattr(config, "smoothing_target", "all_splits")
    smooth_y = bool(getattr(config, "smoothing_y", True))
    if (
        config.enable_smoothing
        and config.smoothing_k > 1
        and smoothing_target != "none"
    ):
        if smoothing_target in {"train_only", "all_splits"}:
            X_train, Y_train, cell_train = _apply_knn_smoothing(
                X_train,
                Y_train,
                cell_train,
                group_size=config.smoothing_k,
                n_components=config.smoothing_pca_components,
                random_state=config.random_state,
                split_label="train",
                smooth_y=smooth_y,
            )
        if smoothing_target == "all_splits":
            X_val, Y_val, cell_val = _apply_knn_smoothing(
                X_val,
                Y_val,
                cell_val,
                group_size=config.smoothing_k,
                n_components=config.smoothing_pca_components,
                random_state=config.random_state + 1,
                split_label="val",
                smooth_y=smooth_y,
            )
            X_test, Y_test, cell_test = _apply_knn_smoothing(
                X_test,
                Y_test,
                cell_test,
                group_size=config.smoothing_k,
                n_components=config.smoothing_pca_components,
                random_state=config.random_state + 2,
                split_label="test",
                smooth_y=smooth_y,
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

    prepared_cellwise = PreparedCellwiseData(
        splits=splits, feature_scaler=feature_scaler, target_scaler=target_scaler
    )

    return prepared_cellwise


def train_model_for_gene(
    dataset,
    model_name: str,
    config: TrainingConfig,
    artifacts_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    wandb_run: Optional[Any] = None,
    global_knn: Optional[GlobalSplitKNN] = None,
    progress_label: Optional[str] = None,
) -> ModelResult:
    _seed_everything(config.random_state)
    runtime_config = config
    if model_name.lower() in {
        "cnn",
        "resnet",
        "rnn",
        "lstm",
        "transformer",
        "mlp",
        "dcn",
        "graph",
    } and getattr(config, "per_gene_torch_stability_profile", False):
        _pge = getattr(config, "per_gene_torch_epochs", None)
        _pgm = getattr(config, "per_gene_torch_min_epochs", None)
        runtime_config = replace(
            config,
            learning_rate=min(
                config.learning_rate, config.per_gene_torch_learning_rate
            ),
            weight_decay=max(config.weight_decay, config.per_gene_torch_weight_decay),
            early_stopping_patience=max(
                config.early_stopping_patience,
                config.per_gene_torch_early_stopping_patience,
            ),
            **({} if _pge is None else {"epochs": _pge}),
            **({} if _pgm is None else {"min_epochs_before_early_stopping": _pgm}),
        )
        _LOG.info(
            "Per-gene torch stability profile active | model=%s | lr=%.6g | weight_decay=%.6g | "
            "early_stopping_patience=%d | epochs=%d | min_epochs=%d",
            model_name,
            runtime_config.learning_rate,
            runtime_config.weight_decay,
            runtime_config.early_stopping_patience,
            runtime_config.epochs,
            runtime_config.min_epochs_before_early_stopping,
        )

    dataset_key = str(
        getattr(dataset.gene, "gene_id", "")
        or getattr(dataset.gene, "gene_name", "unknown")
    )
    cache_key = _config_cache_key(
        config,
        scope="gene",
        dataset_key=dataset_key,
        use_global_knn=(global_knn is not None),
    )
    cache_dict = getattr(dataset, "prepared_cache", {})
    if cache_dict is None:
        cache_dict = {}
        setattr(dataset, "prepared_cache", cache_dict)
    prepared: PreparedData
    if cache_key in cache_dict:
        prepared = cache_dict[cache_key]  # type: ignore[assignment]
        if hasattr(dataset, "gene"):
            _LOG.info(
                "Reusing cached prepared data for gene %s",
                getattr(dataset.gene, "gene_name", "unknown"),
            )
    else:
        prepared = prepare_data(
            dataset, config, cache_dir=cache_dir, global_knn=global_knn
        )
        cache_dict[cache_key] = prepared
    splits = prepared.splits
    fallback_reasons: List[str] = []

    cv_metrics: List[FoldMetrics] = []
    if runtime_config.k_folds <= 1:
        _LOG.debug(
            "Cross-validation disabled for gene %s | model=%s | using only the train/val/test split",
            dataset.gene.gene_name,
            model_name,
        )
    else:
        cv_groups = np.asarray(splits.group_labels_train)
        unique_groups = np.unique(cv_groups)
        use_group_kfold = (
            bool(runtime_config.group_key)
            and unique_groups.size >= runtime_config.k_folds
        )
        if use_group_kfold:
            kf = GroupKFold(n_splits=runtime_config.k_folds)
            splitter = kf.split(splits.X_train, splits.y_train, cv_groups)
        else:
            if runtime_config.group_key:
                _LOG.warning(
                    "Insufficient unique groups (%d) for GroupKFold (k=%d); falling back to standard KFold",
                    unique_groups.size,
                    runtime_config.k_folds,
                )
            kf = KFold(
                n_splits=runtime_config.k_folds,
                shuffle=True,
                random_state=runtime_config.random_state,
            )
            splitter = kf.split(splits.X_train)

        for fold_idx, (train_idx, val_idx) in enumerate(splitter, start=1):
            X_train_source = (
                splits.X_train_raw if splits.X_train_raw is not None else splits.X_train
            )
            y_train_source = (
                splits.y_train_raw if splits.y_train_raw is not None else splits.y_train
            )

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
                runtime_config,
                artifacts_dir=fold_artifacts_dir,
            )
            if isinstance(model, TorchModelBundle):
                sequence_kwargs = _sequence_distance_kwargs(
                    dataset,
                    runtime_config,
                    model.reshape,
                )
                fallback_reason = _torch_dummy_reason_single_output(X_tr, y_tr)
                if fallback_reason:
                    fallback_reasons.append(
                        f"{model_name}:{dataset.gene.gene_name}:torch CV fold fallback ({fallback_reason})"
                    )
                    _LOG.warning(
                        "Using DummyRegressor fallback for torch CV fold | model=%s | gene=%s | reason=%s",
                        model_name,
                        dataset.gene.gene_name,
                        fallback_reason,
                    )
                    _, preds = _fit_dummy_single_output(X_tr, y_tr, X_va)
                else:
                    try:
                        _, preds, _ = _fit_torch_model(
                            model,
                            X_tr,
                            y_tr,
                            X_va,
                            y_va,
                            runtime_config,
                            model_name=model_name,
                            target_scaler=fold_target_scaler,
                            capture_history=False,
                            wandb_run=None,
                            **sequence_kwargs,
                        )
                        preds = np.asarray(preds).ravel()
                        if not np.isfinite(preds).all():
                            raise ValueError(
                                "non-finite predictions returned by torch model"
                            )
                    except Exception as exc:
                        fallback_reasons.append(
                            f"{model_name}:{dataset.gene.gene_name}:torch CV fold failed ({exc})"
                        )
                        _LOG.warning(
                            "Torch CV fold failed; using DummyRegressor fallback | model=%s | gene=%s | error=%s",
                            model_name,
                            dataset.gene.gene_name,
                            exc,
                        )
                        _, preds = _fit_dummy_single_output(X_tr, y_tr, X_va)
            else:
                estimator = clone(model)
                estimator = _fit_single_output_estimator(
                    estimator,
                    X_tr,
                    y_tr,
                    model_name=model_name,
                    gene_name=dataset.gene.gene_name,
                    fallback_reasons=fallback_reasons,
                )
                preds = estimator.predict(X_va)
            scaler_for_metrics = (
                fold_target_scaler
                if fold_target_scaler is not None
                else prepared.target_scaler
            )
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
        runtime_config,
        artifacts_dir=final_artifacts_dir,
    )
    if isinstance(model, TorchModelBundle):
        sequence_kwargs = _sequence_distance_kwargs(
            dataset,
            runtime_config,
            model.reshape,
        )

        fallback_reason = _torch_dummy_reason_single_output(
            splits.X_train, splits.y_train
        )
        if fallback_reason:
            fallback_reasons.append(
                f"{model_name}:{dataset.gene.gene_name}:final torch fit fallback ({fallback_reason})"
            )
            _LOG.warning(
                "Using DummyRegressor fallback for final torch fit | model=%s | gene=%s | reason=%s",
                model_name,
                dataset.gene.gene_name,
                fallback_reason,
            )
            fitted_model, pred_train = _fit_dummy_single_output(
                splits.X_train, splits.y_train, splits.X_train
            )
            pred_val = fitted_model.predict(np.asarray(splits.X_val))
            pred_test = fitted_model.predict(np.asarray(splits.X_test))
            pred_val = np.asarray(pred_val).ravel()
            pred_test = np.asarray(pred_test).ravel()
            history = None
        else:
            try:
                fitted_model, _, history = _fit_torch_model(
                    model,
                    splits.X_train,
                    splits.y_train,
                    splits.X_val,
                    splits.y_val,
                    runtime_config,
                    model_name=model_name,
                    target_scaler=prepared.target_scaler,
                    capture_history=True,
                    wandb_run=None,
                    **sequence_kwargs,
                )
                pred_train = np.asarray(
                    _predict_torch(
                        fitted_model,
                        model.reshape,
                        runtime_config,
                        splits.X_train,
                        **sequence_kwargs,
                    )
                ).ravel()
                pred_val = np.asarray(
                    _predict_torch(
                        fitted_model,
                        model.reshape,
                        runtime_config,
                        splits.X_val,
                        **sequence_kwargs,
                    )
                ).ravel()
                pred_test = np.asarray(
                    _predict_torch(
                        fitted_model,
                        model.reshape,
                        runtime_config,
                        splits.X_test,
                        **sequence_kwargs,
                    )
                ).ravel()
                if not (
                    np.isfinite(pred_train).all()
                    and np.isfinite(pred_val).all()
                    and np.isfinite(pred_test).all()
                ):
                    raise ValueError("non-finite predictions returned by torch model")
            except Exception as exc:
                fallback_reasons.append(
                    f"{model_name}:{dataset.gene.gene_name}:torch final fit failed ({exc})"
                )
                _LOG.warning(
                    "Torch final fit failed; using DummyRegressor fallback | model=%s | gene=%s | error=%s",
                    model_name,
                    dataset.gene.gene_name,
                    exc,
                )
                fitted_model, pred_train = _fit_dummy_single_output(
                    splits.X_train, splits.y_train, splits.X_train
                )
                pred_val = fitted_model.predict(np.asarray(splits.X_val))
                pred_test = fitted_model.predict(np.asarray(splits.X_test))
                pred_val = np.asarray(pred_val).ravel()
                pred_test = np.asarray(pred_test).ravel()
                history = None
    else:
        estimator = clone(model)
        estimator = _fit_single_output_estimator(
            estimator,
            splits.X_train,
            splits.y_train,
            model_name=model_name,
            gene_name=dataset.gene.gene_name,
            fallback_reasons=fallback_reasons,
        )
        pred_train = estimator.predict(splits.X_train)
        pred_val = estimator.predict(splits.X_val)
        pred_test = estimator.predict(splits.X_test)
        fitted_model = estimator  # type: ignore
        history = None

    y_train_true = _unscale_targets(prepared.target_scaler, splits.y_train)
    y_val_true = _unscale_targets(prepared.target_scaler, splits.y_val)
    y_test_true = _unscale_targets(prepared.target_scaler, splits.y_test)
    pred_train_unscaled = _unscale_targets(prepared.target_scaler, pred_train)
    pred_val_unscaled = _unscale_targets(prepared.target_scaler, pred_val)
    pred_test_unscaled = _unscale_targets(prepared.target_scaler, pred_test)
    pred_train_unscaled, pred_val_unscaled, pred_test_unscaled = (
        _postprocess_unscaled_predictions(
            runtime_config,
            X_train=splits.X_train,
            X_val=splits.X_val,
            X_test=splits.X_test,
            y_train_true=y_train_true,
            y_val_true=y_val_true,
            pred_train=pred_train_unscaled,
            pred_val=pred_val_unscaled,
            pred_test=pred_test_unscaled,
        )
    )

    train_metrics = regression_metrics(y_train_true, pred_train_unscaled)
    val_metrics = regression_metrics(y_val_true, pred_val_unscaled)
    test_metrics = regression_metrics(y_test_true, pred_test_unscaled)

    if progress_label:
        _LOG.info(
            "Final metrics | progress=%s | gene=%s | model=%s | train_R2=%.4f | val_R2=%.4f | test_R2=%.4f",
            progress_label,
            dataset.gene.gene_name,
            model_name,
            train_metrics.get("r2", float("nan")),
            val_metrics.get("r2", float("nan")),
            test_metrics.get("r2", float("nan")),
        )
    else:
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
            "train": (
                splits.cell_ids_train,
                y_train_true,
                pred_train_unscaled,
            ),
            "val": (
                splits.cell_ids_val,
                y_val_true,
                pred_val_unscaled,
            ),
            "test": (
                splits.cell_ids_test,
                y_test_true,
                pred_test_unscaled,
            ),
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
        data_counts=dict(splits.metadata),
        used_fallback=bool(fallback_reasons),
        fallback_reasons=fallback_reasons,
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
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Estimate global feature importance via input gradients with permutation fallback.

    Returns mean absolute gradients (for ranking) and mean signed gradients (for directionality).
    """

    X_ref = np.asarray(X_reference, dtype=np.float32)
    y_ref = (
        np.asarray(y_reference, dtype=np.float64) if y_reference is not None else None
    )

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

    totals_abs = np.zeros(X_ref.shape[1], dtype=np.float64)
    totals_signed = np.zeros(X_ref.shape[1], dtype=np.float64)
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

            grad_np = grads.detach().cpu().numpy()
            totals_signed += grad_np.sum(axis=0)
            totals_abs += np.abs(grad_np).sum(axis=0)
            count += grad_np.shape[0]
        if count > 0:
            grad_success = True
            return totals_abs / float(count), totals_signed / float(count)
    except Exception:
        grad_success = False

    if not grad_success and y_ref is not None:
        perm = _compute_torch_permutation_importance(
            bundle,
            X_ref,
            y_ref,
            device=device,
            target_scaler=target_scaler,
            max_samples=max_samples,
            batch_size=batch_size,
        )
        if perm is None:
            return None
        return perm, None
    return None


def _compute_torch_shap_importance(
    bundle: TorchModelBundle,
    X_reference: np.ndarray,
    *,
    device: torch.device,
    max_samples: Optional[int] = 500,
    background_samples: int = 100,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Estimate mean absolute and mean signed SHAP values for a torch model (best-effort)."""
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
        mean_abs = np.mean(np.abs(shap_arr), axis=(0, 1))
        mean_signed = np.mean(shap_arr, axis=(0, 1))
        return mean_abs, mean_signed

    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim > 2:
        shap_arr = shap_arr.reshape(shap_arr.shape[0], -1)
    if shap_arr.ndim != 2:
        return None
    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    mean_signed = np.mean(shap_arr, axis=0)
    return mean_abs, mean_signed


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
    wandb_run: Optional[Any] = None,
) -> CellwiseModelResult:
    _seed_everything(config.random_state)
    runtime_config = config
    if config.fast_classical_mode and model_name in _FAST_CLASSICAL_MODELS:
        runtime_config = replace(config)
        runtime_config.enable_smoothing = False
        runtime_config.smoothing_k = 1
        runtime_config.k_folds = min(config.k_folds, 2)
        if model_name == "svr":
            # SVR is especially costly on large single-cell folds; increase
            # pseudobulk aggregation in fast mode to keep runtime bounded.
            runtime_config.pseudobulk_group_size = max(
                config.pseudobulk_group_size,
                FAST_MODE_MIN_PSEUDOBULK_GROUP_SIZE,
            )
        _LOG.info(
            "Fast classical mode active | model=%s | k_folds=%d | smoothing=off | pseudobulk_group_size=%d | group_key=%s",
            model_name,
            runtime_config.k_folds,
            runtime_config.pseudobulk_group_size,
            runtime_config.group_key if runtime_config.group_key else "none",
        )

    cache_key = _cellwise_cache_key(dataset, runtime_config)
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
        prepared = prepare_cellwise_data(dataset, runtime_config, cache_dir=cache_dir)
        cache_dict[cache_key] = prepared
    splits = prepared.splits
    gene_names = [gene.gene_name for gene in dataset.genes]
    target_dim = dataset.y.shape[1]
    catboost_artifacts_root = (
        artifacts_dir
        if artifacts_dir is not None and model_name == "catboost"
        else None
    )

    _LOG.info(
        "Training multi-output model | model=%s | genes=%d | train_samples=%d | features=%d | targets=%d",
        model_name,
        len(gene_names),
        splits.X_train.shape[0],
        splits.X_train.shape[1],
        target_dim,
    )
    _log_resource_snapshot(f"train_multi_output:{model_name}:start")

    cv_metrics: List[FoldMetrics] = []
    if runtime_config.k_folds <= 1:
        _LOG.info(
            "Cross-validation disabled | model=%s | using only the train/val/test split",
            model_name,
        )
    else:
        cv_groups = np.asarray(splits.group_labels_train)
        unique_groups = np.unique(cv_groups)
        use_group_kfold = (
            bool(runtime_config.group_key)
            and unique_groups.size >= runtime_config.k_folds
        )
        if use_group_kfold:
            kf = GroupKFold(n_splits=runtime_config.k_folds)
            splitter = kf.split(splits.X_train, splits.y_train, cv_groups)
        else:
            if runtime_config.group_key:
                _LOG.warning(
                    "Insufficient unique groups (%d) for GroupKFold (k=%d); falling back to KFold",
                    unique_groups.size,
                    runtime_config.k_folds,
                )
            kf = KFold(
                n_splits=runtime_config.k_folds,
                shuffle=True,
                random_state=runtime_config.random_state,
            )
            splitter = kf.split(splits.X_train)

        for fold_idx, (train_idx, val_idx) in enumerate(splitter, start=1):
            fold_start = time.perf_counter()
            X_train_source = (
                splits.X_train_raw if splits.X_train_raw is not None else splits.X_train
            )
            y_train_source = (
                splits.y_train_raw if splits.y_train_raw is not None else splits.y_train
            )

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
                runtime_config.k_folds,
                model_name,
                X_tr.shape[0],
                X_va.shape[0],
            )
            _log_resource_snapshot(
                f"train_multi_output:{model_name}:fold{fold_idx}:start"
            )
            fold_artifacts_dir = None
            if catboost_artifacts_root is not None:
                fold_artifacts_dir = catboost_artifacts_root / f"cv_fold_{fold_idx}"
            model = build_model(
                model_name,
                dataset.X.shape[1],
                runtime_config,
                output_dim=target_dim,
                artifacts_dir=fold_artifacts_dir,
                feature_block_indices=getattr(dataset, "feature_block_indices", None),
            )
            if isinstance(model, TorchModelBundle):
                _, preds, _ = _fit_torch_model(
                    model,
                    X_tr,
                    y_tr,
                    X_va,
                    y_va,
                    runtime_config,
                    model_name=model_name,
                    target_scaler=fold_target_scaler,
                    capture_history=False,
                    wandb_run=None,
                )
            else:
                estimator = clone(model)
                estimator.fit(X_tr, y_tr)
                preds = estimator.predict(X_va)

            scaler_for_metrics = (
                fold_target_scaler
                if fold_target_scaler is not None
                else prepared.target_scaler
            )
            y_val_true = _ensure_2d(_unscale_targets(scaler_for_metrics, y_va))
            y_val_pred = _ensure_2d(_unscale_targets(scaler_for_metrics, preds))
            agg_metrics, _ = _compute_multi_metrics(y_val_true, y_val_pred, gene_names)
            cv_metrics.append(FoldMetrics(fold=fold_idx, metrics=agg_metrics))
            duration = time.perf_counter() - fold_start
            _LOG.info(
                "Completed CV fold %d/%d | model=%s | mean_R2=%.4f | mean_RMSE=%.4f | %.2fs",
                fold_idx,
                runtime_config.k_folds,
                model_name,
                agg_metrics.get("r2", float("nan")),
                agg_metrics.get("rmse", float("nan")),
                duration,
            )
            _log_resource_snapshot(
                f"train_multi_output:{model_name}:fold{fold_idx}:end"
            )

    final_artifacts_dir = None
    if catboost_artifacts_root is not None:
        final_artifacts_dir = catboost_artifacts_root / "final_fit"
    model = build_model(
        model_name,
        dataset.X.shape[1],
        runtime_config,
        output_dim=target_dim,
        artifacts_dir=final_artifacts_dir,
        feature_block_indices=getattr(dataset, "feature_block_indices", None),
    )
    fit_start = time.perf_counter()
    if isinstance(model, TorchModelBundle):
        fitted_model, _, history = _fit_torch_model(
            model,
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            runtime_config,
            model_name=model_name,
            target_scaler=prepared.target_scaler,
            capture_history=True,
            wandb_run=wandb_run,
        )
        pred_train = _predict_torch(
            fitted_model,
            model.reshape,
            runtime_config,
            splits.X_train,
            output_dim=target_dim,
        )
        pred_val = _predict_torch(
            fitted_model,
            model.reshape,
            runtime_config,
            splits.X_val,
            output_dim=target_dim,
        )
        pred_test = _predict_torch(
            fitted_model,
            model.reshape,
            runtime_config,
            splits.X_test,
            output_dim=target_dim,
        )
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
    feature_importance_mean_signed: Optional[np.ndarray] = None
    feature_importance_method: Optional[str] = None
    shap_importance_mean: Optional[np.ndarray] = None
    shap_value_mean_signed: Optional[np.ndarray] = None
    shap_importance_method: Optional[str] = None
    if isinstance(model, TorchModelBundle) and config.enable_feature_importance:
        fi_start = time.perf_counter()
        fi_max_samples = config.feature_importance_samples
        fi_batch_size = config.feature_importance_batch_size
        try:
            device = _select_device(config.device_preference)
            fi_result = _compute_torch_feature_importance(
                model,
                splits.X_val,
                splits.y_val,
                device=device,
                max_samples=(
                    fi_max_samples
                    if fi_max_samples is not None
                    else splits.X_val.shape[0]
                ),
                batch_size=fi_batch_size,
                target_scaler=prepared.target_scaler,
            )
            if fi_result is not None:
                feature_importances, feature_importance_mean_signed = fi_result
                feature_importance_method = "input_gradient_abs_mean"
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            _LOG.warning(
                "Failed to compute feature importances for %s: %s", model_name, exc
            )
            feature_importances = None
            feature_importance_mean_signed = None
            feature_importance_method = None
        finally:
            fi_elapsed = time.perf_counter() - fi_start
            eff_samples = int(
                min(splits.X_val.shape[0], fi_max_samples)
                if fi_max_samples is not None
                else splits.X_val.shape[0]
            )
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
            shap_result = _compute_torch_shap_importance(
                model,
                splits.X_val,
                device=device,
                max_samples=config.shap_max_samples,
                background_samples=config.shap_background_samples,
            )
            if shap_result is not None:
                shap_importance_mean, shap_value_mean_signed = shap_result
                shap_importance_method = "shap_gradient_explainer_mean_abs"
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            _LOG.warning(
                "Failed to compute SHAP importances for %s: %s", model_name, exc
            )
            shap_importance_mean = None
            shap_value_mean_signed = None
            shap_importance_method = None
        finally:
            shap_elapsed = time.perf_counter() - shap_start
            _LOG.info(
                "SHAP completed | model=%s | method=%s | samples=%s | background=%d | %.2fs",
                model_name,
                shap_importance_method or "n/a",
                (
                    config.shap_max_samples
                    if config.shap_max_samples is not None
                    else "all"
                ),
                config.shap_background_samples,
                shap_elapsed,
            )

    y_train_true = _ensure_2d(_unscale_targets(prepared.target_scaler, splits.y_train))
    y_val_true = _ensure_2d(_unscale_targets(prepared.target_scaler, splits.y_val))
    y_test_true = _ensure_2d(_unscale_targets(prepared.target_scaler, splits.y_test))

    y_train_pred = _ensure_2d(_unscale_targets(prepared.target_scaler, pred_train))
    y_val_pred = _ensure_2d(_unscale_targets(prepared.target_scaler, pred_val))
    y_test_pred = _ensure_2d(_unscale_targets(prepared.target_scaler, pred_test))
    y_train_pred, y_val_pred, y_test_pred = _postprocess_unscaled_predictions(
        runtime_config,
        X_train=splits.X_train,
        X_val=splits.X_val,
        X_test=splits.X_test,
        y_train_true=y_train_true,
        y_val_true=y_val_true,
        pred_train=y_train_pred,
        pred_val=y_val_pred,
        pred_test=y_test_pred,
    )
    y_train_pred = _ensure_2d(y_train_pred)
    y_val_pred = _ensure_2d(y_val_pred)
    y_test_pred = _ensure_2d(y_test_pred)

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
        feature_importance_mean_signed=feature_importance_mean_signed,
        feature_names=getattr(dataset, "feature_names", None),
        feature_importance_method=feature_importance_method,
        shap_importance_mean=shap_importance_mean,
        shap_value_mean_signed=shap_value_mean_signed,
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
    model_name: Optional[str] = None,
    target_scaler: Optional[StandardScaler | MinMaxScaler] = None,
    capture_history: bool = False,
    wandb_run: Optional[Any] = None,
    sequence_offsets: Optional[np.ndarray] = None,
    sequence_offset_features: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, np.ndarray, Optional[List[Dict[str, float]]]]:
    device = _select_device(config.device_preference)
    model = bundle.model.to(device)
    model = _wrap_model_for_multi_gpu(model, device)

    X_train = _sanitize_numeric_array("X_train", X_train)
    X_val = _sanitize_numeric_array("X_val", X_val)
    y_train = _sanitize_numeric_array("y_train", y_train)
    y_val = _sanitize_numeric_array("y_val", y_val)

    y_train_arr = np.asarray(y_train)
    target_dim = y_train_arr.shape[1] if y_train_arr.ndim > 1 else 1

    train_ds = _make_dataset(
        bundle.reshape,
        X_train,
        y_train,
        sequence_offsets=sequence_offsets,
        sequence_offset_features=sequence_offset_features,
    )
    val_ds = _make_dataset(
        bundle.reshape,
        X_val,
        y_val,
        sequence_offsets=sequence_offsets,
        sequence_offset_features=sequence_offset_features,
    )

    # Per-model batch size: look up model_name in per_model_batch_size, fall back to batch_size
    _per_model_bs = getattr(config, "per_model_batch_size", {})
    _base_bs = (
        _per_model_bs.get(model_name, config.batch_size)
        if model_name and _per_model_bs
        else config.batch_size
    )
    _bs_config = (
        replace(config, batch_size=_base_bs)
        if _base_bs != config.batch_size
        else config
    )
    batch_size = _effective_batch_size(_bs_config, target_dim)

    # Preload tensors to GPU when they fit in available VRAM to eliminate per-batch H2D transfers
    _preloaded = False
    if getattr(config, "preload_to_device", False) and device.type == "cuda":
        _ds_bytes = sum(
            t.element_size() * t.numel() for t in train_ds.tensors + val_ds.tensors
        )
        try:
            _free, _ = torch.cuda.mem_get_info(device)
            if _ds_bytes < _free * 0.5:
                train_ds = TensorDataset(
                    *[t.to(device, non_blocking=True) for t in train_ds.tensors]
                )
                val_ds = TensorDataset(
                    *[t.to(device, non_blocking=True) for t in val_ds.tensors]
                )
                _preloaded = True
        except Exception:
            pass

    _pin = device.type == "cuda" and not _preloaded
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=_pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=_pin
    )
    track_history = capture_history and config.track_history
    history: List[Dict[str, float]] = []
    train_eval_loader: Optional[DataLoader] = None
    if track_history:
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=_pin,
        )

    criterion = nn.MSELoss()
    pearson_loss_weight = float(getattr(config, "torch_pearson_loss_weight", 0.0))

    def _batch_pearson_loss(
        preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        preds_centered = preds - preds.mean(dim=0, keepdim=True)
        targets_centered = targets - targets.mean(dim=0, keepdim=True)
        cov = torch.sum(preds_centered * targets_centered, dim=0)
        pred_norm = torch.sqrt(torch.sum(preds_centered * preds_centered, dim=0) + eps)
        target_norm = torch.sqrt(
            torch.sum(targets_centered * targets_centered, dim=0) + eps
        )
        corr = cov / (pred_norm * target_norm + eps)
        return 1.0 - corr.clamp(-1.0, 1.0).mean()

    def _compute_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = criterion(outputs, targets)
        if pearson_loss_weight <= 0.0:
            return mse
        return mse + pearson_loss_weight * _batch_pearson_loss(outputs, targets)

    _opt_cls = (
        torch.optim.AdamW
        if getattr(config, "optimizer", "adamw") == "adamw"
        else torch.optim.Adam
    )
    optimizer = _opt_cls(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    accumulation_steps = max(1, int(config.gradient_accumulation_steps))
    updates_per_epoch = max(
        1, math.ceil(max(1, len(train_loader)) / accumulation_steps)
    )

    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    total_scheduler_steps = 0
    warmup_scheduler_steps = 0
    if config.lr_scheduler == "cosine":
        total_scheduler_steps = max(1, updates_per_epoch * max(1, int(config.epochs)))
        warmup_from_epochs = max(0, int(config.warmup_epochs)) * updates_per_epoch
        warmup_from_ratio = int(
            round(total_scheduler_steps * float(config.warmup_ratio))
        )
        warmup_scheduler_steps = (
            warmup_from_epochs if warmup_from_epochs > 0 else warmup_from_ratio
        )
        warmup_scheduler_steps = min(
            max(0, warmup_scheduler_steps), max(0, total_scheduler_steps - 1)
        )
        min_lr_ratio = float(config.min_lr_ratio)

        def _lr_lambda(step: int) -> float:
            if step < warmup_scheduler_steps:
                return float(step + 1) / float(max(1, warmup_scheduler_steps))
            progress_denom = max(1, total_scheduler_steps - warmup_scheduler_steps - 1)
            progress = min(
                1.0, float(step - warmup_scheduler_steps) / float(progress_denom)
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    use_amp = device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)

    best_state = None
    best_val = float("inf")
    patience = config.early_stopping_patience
    min_epochs_before_early_stop = max(
        1, int(getattr(config, "min_epochs_before_early_stopping", 1))
    )
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
            return np.empty((0, target_dim), dtype=np.float32), np.empty(
                (0, target_dim), dtype=np.float32
            )
        pred_arr = np.concatenate(preds, axis=0)
        truth_arr = np.concatenate(truths, axis=0)
        if pred_arr.ndim == 1:
            pred_arr = pred_arr.reshape(-1, 1)
        if truth_arr.ndim == 1:
            truth_arr = truth_arr.reshape(-1, 1)
        return pred_arr, truth_arr

    def _compute_metric_summary(
        y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray
    ) -> Dict[str, float]:
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
        "Starting training | model=%s | device=%s | epochs=%d | batch_size=%d | grad_accum_steps=%d | optimizer_step_batch_size=%d | lr_scheduler=%s | pearson_loss_weight=%.4f | early_stopping_patience=%d | min_epochs_before_early_stopping=%d",
        type(model).__name__,
        device.type,
        config.epochs,
        batch_size,
        accumulation_steps,
        batch_size * accumulation_steps,
        config.lr_scheduler,
        pearson_loss_weight,
        patience,
        min_epochs_before_early_stop,
    )
    if scheduler is not None:
        _LOG.info(
            "LR scheduler enabled | type=cosine | warmup_steps=%d | total_steps=%d | min_lr_ratio=%.4f",
            warmup_scheduler_steps,
            total_scheduler_steps,
            config.min_lr_ratio,
        )
    _log_gpu_memory_snapshot("Before training start")

    for epoch in range(config.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        train_loss_accum = 0.0
        train_samples = 0
        nan_detected = False
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            with _amp_autocast(device.type, use_amp):
                outputs = model(batch_x)
                loss = _compute_loss(outputs, batch_y)
            if not torch.isfinite(loss):
                _LOG.warning(
                    "Non-finite training loss detected at epoch %d; stopping early.",
                    epoch + 1,
                )
                nan_detected = True
                break
            loss_for_backward = loss / accumulation_steps
            scaler.scale(loss_for_backward).backward()
            should_step = ((batch_idx + 1) % accumulation_steps == 0) or (
                (batch_idx + 1) == len(train_loader)
            )
            if should_step:
                if config.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                did_optimizer_step = True
                if use_amp and hasattr(scaler, "get_scale"):
                    prev_scale = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    # When GradScaler skips an update due to inf/NaN grads, scale decreases.
                    did_optimizer_step = float(scaler.get_scale()) >= prev_scale
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and did_optimizer_step:
                    scheduler.step()
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
                    val_loss = _compute_loss(preds, val_y)
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
            best_state = _snapshot_model_state(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if (epoch + 1) >= min_epochs_before_early_stop:
                    should_stop = True
                    _LOG.info(
                        "Early stopping triggered at epoch %d/%d | best_val=%.6f | current_val=%.6f | epochs_no_improve=%d",
                        epoch + 1,
                        config.epochs,
                        best_val,
                        mean_val,
                        epochs_no_improve,
                    )
                else:
                    _LOG.info(
                        "Early stopping patience reached at epoch %d but deferred until min_epochs_before_early_stopping=%d",
                        epoch + 1,
                        min_epochs_before_early_stop,
                    )

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

            train_metrics = _compute_metric_summary(
                train_true_scaled, train_pred_scaled
            )
            val_metrics = _compute_metric_summary(val_true_scaled, val_pred_scaled)

            entry: Dict[str, float] = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss_mean),
                "val_loss": float(mean_val),
                "effective_batch_size": float(batch_size * accumulation_steps),
                "micro_batch_size": float(batch_size),
                "grad_accum_steps": float(accumulation_steps),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if process is not None:
                try:
                    entry["cpu_percent"] = float(process.cpu_percent(interval=None))
                except Exception:
                    pass
                try:
                    if "gpu_util_pct" in locals() and gpu_util_pct is not None:
                        entry["rss_gib"] = float(torch.cuda.memory_allocated()) / (
                            1024**3
                        )
                except Exception:
                    pass
                try:
                    entry["thread_count"] = float(process.num_threads())
                except Exception:
                    pass
            if device.type == "cuda":
                entry["gpu_alloc_mb"] = float(torch.cuda.memory_allocated() / (1024**2))
                entry["gpu_reserved_mb"] = float(
                    torch.cuda.memory_reserved() / (1024**2)
                )
                entry["gpu_peak_alloc_mb"] = float(
                    torch.cuda.max_memory_allocated() / (1024**2)
                )
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
            if wandb_run is not None:
                streamed_metrics = {
                    "training/train_loss": entry["train_loss"],
                    "training/val_loss": entry["val_loss"],
                    "training/lr": entry["lr"],
                }
                for metric_name in config.history_metrics:
                    key = metric_name.lower()
                    if key in {"loss", "mae", "mse"}:
                        continue
                    for split in ("train", "val"):
                        metric_key = f"{split}_{key}"
                        if metric_key in entry:
                            streamed_metrics[f"training/{metric_key}"] = entry[
                                metric_key
                            ]
                wandb_log_metrics(wandb_run, streamed_metrics, step=epoch + 1)

        gpu_peak_mb = None
        if device.type == "cuda":
            gpu_peak_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

        if config.track_history and history:
            recent = history[-1]
            log_msg = (
                "Epoch %d/%d | model=%s | batch_size=%d | train_loss=%.6f | val_loss=%.6f"
                % (
                    epoch + 1,
                    config.epochs,
                    type(model).__name__,
                    batch_size * accumulation_steps,
                    recent.get("train_loss", float("nan")),
                    recent.get("val_loss", float("nan")),
                )
            )
            log_msg += f" | lr={optimizer.param_groups[0]['lr']:.6g}"
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
                    batch_size * accumulation_steps,
                    train_loss_accum / max(train_samples, 1),
                    mean_val,
                )
            )
            log_msg += f" | lr={optimizer.param_groups[0]['lr']:.6g}"
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
    preds = _predict_torch(
        model,
        bundle.reshape,
        config,
        X_val,
        output_dim=target_dim,
        sequence_offsets=sequence_offsets,
        sequence_offset_features=sequence_offset_features,
    )
    return model.cpu(), preds, history if history else None


def _predict_torch(
    model: nn.Module,
    reshape: str,
    config: TrainingConfig,
    X: np.ndarray,
    output_dim: int = 1,
    *,
    sequence_offsets: Optional[np.ndarray] = None,
    sequence_offset_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    device = _select_device(config.device_preference)
    model = model.to(device)
    if output_dim > 1:
        placeholder = np.zeros((X.shape[0], output_dim), dtype=np.float32)
    else:
        placeholder = np.zeros(X.shape[0], dtype=np.float32)
    ds = _make_dataset(
        reshape,
        X,
        placeholder,
        sequence_offsets=sequence_offsets,
        sequence_offset_features=sequence_offset_features,
    )
    batch_size = _effective_batch_size(config, output_dim)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda"
    )
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
    result = _sanitize_numeric_array("torch_predictions", result)
    if result.ndim == 2 and result.shape[1] == 1:
        return result.ravel()
    return result


def _make_dataset(
    reshape: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    sequence_offsets: Optional[np.ndarray] = None,
    sequence_offset_features: Optional[np.ndarray] = None,
) -> TensorDataset:
    if reshape == "sequence":
        # By convention sequence inputs are channels-first for conv stems:
        # (batch, channels, seq_len). Channel 0 is the accessibility signal.
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

        # Optional: distance-to-TSS features for peak-basis sequence models.
        # These are appended as additional channels (batch, extra_channels, seq_len).
        if sequence_offset_features is not None:
            feats = np.asarray(sequence_offset_features, dtype=np.float32)
            if feats.ndim == 1:
                feats = feats.reshape(-1, 1)
            if feats.shape[0] != X_tensor.shape[-1]:
                raise ValueError(
                    f"sequence_offset_features length {feats.shape[0]} does not match sequence length {X_tensor.shape[-1]}"
                )
            # feats: (seq_len, channels) -> (1, channels, seq_len) -> expand batch
            feat_tensor = (
                torch.tensor(feats.T, dtype=torch.float32)
                .view(1, feats.shape[1], feats.shape[0])
                .expand(X_tensor.shape[0], -1, -1)
            )
            X_tensor = torch.cat([X_tensor, feat_tensor], dim=1)
        elif sequence_offsets is not None:
            offsets = np.asarray(sequence_offsets, dtype=np.float32).ravel()
            if offsets.size != X_tensor.shape[-1]:
                raise ValueError(
                    f"sequence_offsets length {offsets.size} does not match sequence length {X_tensor.shape[-1]}"
                )
            offset_tensor = (
                torch.tensor(offsets, dtype=torch.float32)
                .view(1, 1, -1)
                .expand(X_tensor.shape[0], -1, -1)
            )
            X_tensor = torch.cat([X_tensor, offset_tensor], dim=1)
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


def _unscale_targets(
    scaler: Optional[StandardScaler | MinMaxScaler], values: np.ndarray
) -> np.ndarray:
    arr = np.asarray(values)
    if scaler is None:
        return arr
    if arr.ndim == 1:
        inv = scaler.inverse_transform(arr.reshape(-1, 1))
        return inv.ravel()
    return scaler.inverse_transform(arr)


def _apply_prediction_floor(config: TrainingConfig, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    floor = getattr(config, "prediction_min_value", 0.0)
    if floor is not None:
        arr = np.maximum(arr, float(floor))
    return arr


def _fit_zero_expression_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
) -> Optional[object]:
    y_binary = (np.asarray(y_train, dtype=np.float64).ravel() > 0.0).astype(int)
    if y_binary.size < 2 or np.unique(y_binary).size < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        classifier.fit(np.asarray(X_train), y_binary)
        return classifier
    except Exception as exc:
        _LOG.warning(
            "Zero-aware classifier fit failed; skipping zero-aware post-processing: %s",
            exc,
        )
        return None


def _expression_probability(classifier: object, X: np.ndarray) -> Optional[np.ndarray]:
    try:
        probs = classifier.predict_proba(np.asarray(X))
        classes = getattr(classifier, "classes_", None)
        if classes is None:
            return probs[:, -1]
        class_list = list(classes)
        if 1 not in class_list:
            return None
        return probs[:, class_list.index(1)]
    except Exception as exc:
        _LOG.warning("Zero-aware classifier prediction failed; skipping split: %s", exc)
        return None


def _postprocess_unscaled_predictions(
    config: TrainingConfig,
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train_true: np.ndarray,
    y_val_true: np.ndarray,
    pred_train: np.ndarray,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply prediction-space post-processing before metrics/export."""
    train = _ensure_2d(_apply_prediction_floor(config, pred_train))
    val = _ensure_2d(_apply_prediction_floor(config, pred_val))
    test = _ensure_2d(_apply_prediction_floor(config, pred_test))
    y_train = _ensure_2d(np.asarray(y_train_true, dtype=np.float64))
    y_val = _ensure_2d(np.asarray(y_val_true, dtype=np.float64))

    if bool(getattr(config, "enable_zero_aware_predictions", False)):
        threshold = float(getattr(config, "zero_aware_threshold", 0.5))
        mode = str(getattr(config, "zero_aware_mode", "mask"))
        for idx in range(train.shape[1]):
            classifier = _fit_zero_expression_classifier(
                X_train,
                y_train[:, idx],
                random_state=int(getattr(config, "random_state", 42)),
            )
            if classifier is None:
                continue
            for X_split, pred_split in (
                (X_train, train),
                (X_val, val),
                (X_test, test),
            ):
                prob = _expression_probability(classifier, X_split)
                if prob is None:
                    continue
                if mode == "multiply":
                    pred_split[:, idx] = pred_split[:, idx] * prob
                else:
                    pred_split[prob < threshold, idx] = 0.0

    if bool(getattr(config, "enable_prediction_calibration", False)):
        try:
            from sklearn.linear_model import LinearRegression
        except Exception as exc:
            _LOG.warning("Prediction calibration unavailable; skipping: %s", exc)
        else:
            for idx in range(train.shape[1]):
                x_cal = val[:, idx]
                y_cal = y_val[:, idx]
                mask = np.isfinite(x_cal) & np.isfinite(y_cal)
                if mask.sum() < 3 or np.allclose(x_cal[mask], x_cal[mask][0]):
                    continue
                try:
                    calibrator = LinearRegression()
                    calibrator.fit(x_cal[mask].reshape(-1, 1), y_cal[mask])
                    train[:, idx] = calibrator.predict(train[:, idx].reshape(-1, 1))
                    val[:, idx] = calibrator.predict(val[:, idx].reshape(-1, 1))
                    test[:, idx] = calibrator.predict(test[:, idx].reshape(-1, 1))
                except Exception as exc:
                    _LOG.warning(
                        "Prediction calibration failed for output %d; leaving uncalibrated: %s",
                        idx,
                        exc,
                    )

    train = _apply_prediction_floor(config, train)
    val = _apply_prediction_floor(config, val)
    test = _apply_prediction_floor(config, test)
    if np.asarray(pred_train).ndim == 1:
        return train.ravel(), val.ravel(), test.ravel()
    return train, val, test


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
    precomputed_neighbor_idx: Optional[np.ndarray] = None,
    smooth_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-NN smoothing: average each cell with its k-1 nearest neighbors (dataset size unchanged).

    When *precomputed_neighbor_idx* is provided (shape [n_cells, k]) the PCA+kNN step is
    skipped entirely and those indices are used directly.  This is the path taken by per-gene
    mode when a GlobalSplitKNN has been precomputed from the full ATAC matrix.
    """
    if X.size == 0:
        return X, Y, cell_ids

    n_cells = X.shape[0]

    if precomputed_neighbor_idx is not None:
        neighbor_set = precomputed_neighbor_idx
        X_smoothed = np.asarray(X[neighbor_set], dtype=np.float32).mean(axis=1)
        if not smooth_y:
            Y_smoothed = Y
        elif Y.ndim == 1:
            Y_smoothed = np.asarray(Y[neighbor_set], dtype=np.float32).mean(axis=1)
        else:
            Y_smoothed = np.asarray(Y[neighbor_set], dtype=np.float32).mean(axis=1)
        _LOG.info(
            "Smoothing applied to %s split: %d cells (global precomputed kNN, k=%d, smooth_y=%s)",
            split_label,
            n_cells,
            neighbor_set.shape[1],
            smooth_y,
        )
        return X_smoothed, Y_smoothed, cell_ids

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
        _LOG.warning(
            "PCA failed for %s split (%s); skipping smoothing", split_label, exc
        )
        return X, Y, cell_ids

    k_neighbors = min(group_size - 1, n_cells - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    nn.fit(embedding)
    _, neighbor_indices = nn.kneighbors(embedding)

    # Vectorized neighbor averaging for speed on large datasets
    neighbor_set = neighbor_indices[:, : k_neighbors + 1]
    X_smoothed = np.asarray(X[neighbor_set], dtype=np.float32).mean(axis=1)
    if not smooth_y:
        Y_smoothed = Y
    elif Y.ndim == 1:
        Y_smoothed = np.asarray(Y[neighbor_set], dtype=np.float32).mean(axis=1)
    else:
        Y_smoothed = np.asarray(Y[neighbor_set], dtype=np.float32).mean(axis=1)

    elapsed = time.perf_counter() - start_time
    _LOG.info(
        "Smoothing applied to %s split: %d cells (k=%d | components=%d | smooth_y=%s | %.2fs)",
        split_label,
        n_cells,
        k_neighbors,
        components,
        smooth_y,
        elapsed,
    )
    _log_resource_snapshot(f"smoothing:{split_label}:end")

    return X_smoothed, Y_smoothed, cell_ids


def _compute_knn_neighbor_indices(
    X: np.ndarray,
    k: int,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    """Compute kNN neighbor indices via PCA embedding. Returns [n_cells, k] including self."""
    n_cells = X.shape[0]
    if n_cells <= 1 or k <= 1:
        return np.zeros((n_cells, 1), dtype=np.int64)

    k_actual = min(k, n_cells)
    components = max(1, min(n_components, X.shape[1], n_cells - 1))

    scaler = StandardScaler(with_mean=False)
    try:
        X_scaled = scaler.fit_transform(X)
    except Exception:
        X_scaled = X

    try:
        pca = PCA(n_components=components, random_state=random_state)
        embedding = pca.fit_transform(X_scaled)
    except Exception as exc:
        _LOG.warning(
            "PCA failed in _compute_knn_neighbor_indices: %s; using raw features", exc
        )
        embedding = (
            X_scaled if not sp.issparse(X_scaled) else np.asarray(X_scaled.toarray())
        )

    nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
    nn.fit(embedding)
    _, neighbor_indices = nn.kneighbors(embedding)
    return neighbor_indices  # [n_cells, k_actual] — first column is self (distance 0)


def precompute_global_split_knn(
    atac_matrix: "np.ndarray | sp.spmatrix",
    cell_ids: np.ndarray,
    groups: np.ndarray,
    config: TrainingConfig,
) -> GlobalSplitKNN:
    """Precompute train/val/test splits and kNN neighbor indices from the full filtered ATAC matrix.

    The splits produced here use identical parameters (random_state, fractions, group_key) to
    those computed inside prepare_data, so each gene's prepare_data call can use these results
    directly without re-splitting or re-computing PCA+kNN from its 40-bin local window.
    """
    n_cells = len(cell_ids)
    _LOG.info(
        "Precomputing global split kNN | cells=%d | atac_features=%d | k=%d | pca_components=%d",
        n_cells,
        atac_matrix.shape[1],
        config.smoothing_k,
        config.smoothing_pca_components,
    )

    if sp.issparse(atac_matrix):
        est_gib = n_cells * atac_matrix.shape[1] * 4 / 1024**3
        if est_gib > 4.0:
            _LOG.warning(
                "Densifying sparse ATAC for global kNN PCA may require ~%.1f GiB",
                est_gib,
            )
        X_full = atac_matrix.toarray().astype(np.float32)
    else:
        X_full = np.asarray(atac_matrix, dtype=np.float32)

    all_indices = np.arange(n_cells)
    rng_state = config.random_state

    use_group_split = bool(config.group_key)
    if use_group_split and np.unique(groups).size < 2:
        use_group_split = False

    # Mirror the exact same split logic as prepare_data
    if use_group_split:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=config.test_fraction, random_state=rng_state
        )
        try:
            train_val_idx, test_idx = next(splitter.split(all_indices, groups=groups))
        except ValueError:
            use_group_split = False

    if not use_group_split:
        train_val_idx, test_idx = train_test_split(
            all_indices, test_size=config.test_fraction, random_state=rng_state
        )

    group_temp = groups[train_val_idx]
    val_ratio = config.val_fraction / (config.train_fraction + config.val_fraction)

    if use_group_split:
        val_splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=rng_state + 1
        )
        try:
            train_rel, val_rel = next(
                val_splitter.split(train_val_idx, groups=group_temp)
            )
        except ValueError:
            use_group_split = False

    if not use_group_split:
        train_rel, val_rel = train_test_split(
            np.arange(len(train_val_idx)),
            test_size=val_ratio,
            random_state=rng_state + 1,
        )

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    k = config.smoothing_k
    n_pca = config.smoothing_pca_components

    train_neighbor_idx = _compute_knn_neighbor_indices(
        X_full[train_idx], k, n_pca, rng_state
    )
    val_neighbor_idx = _compute_knn_neighbor_indices(
        X_full[val_idx], k, n_pca, rng_state + 1
    )
    test_neighbor_idx = _compute_knn_neighbor_indices(
        X_full[test_idx], k, n_pca, rng_state + 2
    )

    _LOG.info(
        "Global kNN precomputed | train=%d | val=%d | test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    return GlobalSplitKNN(
        train_cell_ids=cell_ids[train_idx],
        val_cell_ids=cell_ids[val_idx],
        test_cell_ids=cell_ids[test_idx],
        train_neighbor_idx=train_neighbor_idx,
        val_neighbor_idx=val_neighbor_idx,
        test_neighbor_idx=test_neighbor_idx,
        group_labels_train=groups[train_idx],
        group_labels_val=groups[val_idx],
        group_labels_test=groups[test_idx],
    )


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
            _LOG.warning(
                "PCA preprocessing failed for %s split (%s); using raw features",
                split_label,
                exc,
            )
            X_for_pca = X
    try:
        pca = PCA(n_components=components, random_state=random_state)
        embedding = pca.fit_transform(X_for_pca)
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning(
            "PCA failed for %s split (%s); skipping pseudobulk", split_label, exc
        )
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
            same_group_targets = [
                g_idx
                for g_idx, label in enumerate(bulk_group_labels)
                if label == idx_group
            ]
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
            Y_bulk.append(
                np.asarray(Y[indices], dtype=np.float64).mean(axis=0, keepdims=True)
            )
        else:
            Y_bulk.append(np.asarray(Y[indices], dtype=np.float64).mean(axis=0))
        bulk_ids.append(f"{split_label}_bulk_{grp_idx:05d}")

    X_out = np.vstack(X_bulk).astype(np.float32)
    if y_is_vector:
        Y_out = np.vstack(Y_bulk).ravel().astype(np.float32)
    else:
        Y_out = np.vstack(Y_bulk).astype(np.float32)
    cell_out = np.asarray(bulk_ids, dtype=str)
    groups_out = np.asarray(
        bulk_group_labels if bulk_group_labels else group_labels_arr, dtype=str
    )

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

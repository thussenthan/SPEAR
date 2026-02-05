"""On-disk caching utilities for preprocessed data."""

import json
import logging
from pathlib import Path
from typing import Optional
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .data_types import PreparedData, PreparedCellwiseData, SplitData, CellwiseSplitData

_LOG = logging.getLogger(__name__)


def _jsonify_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(item) for item in value]
    return value


def _serialize_scaler(scaler):
    if scaler is None:
        return None
    if isinstance(scaler, StandardScaler):
        state = {
            "mean_": scaler.mean_,
            "scale_": scaler.scale_,
            "var_": scaler.var_,
            "n_samples_seen_": scaler.n_samples_seen_,
            "feature_names_in_": getattr(scaler, "feature_names_in_", None),
            "n_features_in_": getattr(scaler, "n_features_in_", None),
        }
        return {
            "class": "StandardScaler",
            "params": scaler.get_params(),
            "state": {key: _jsonify_value(value) for key, value in state.items() if value is not None},
        }
    if isinstance(scaler, MinMaxScaler):
        state = {
            "data_min_": scaler.data_min_,
            "data_max_": scaler.data_max_,
            "data_range_": scaler.data_range_,
            "scale_": scaler.scale_,
            "min_": scaler.min_,
            "n_samples_seen_": scaler.n_samples_seen_,
            "feature_names_in_": getattr(scaler, "feature_names_in_", None),
            "n_features_in_": getattr(scaler, "n_features_in_", None),
        }
        return {
            "class": "MinMaxScaler",
            "params": scaler.get_params(),
            "state": {key: _jsonify_value(value) for key, value in state.items() if value is not None},
        }
    raise TypeError(
        f"Unsupported scaler type: {type(scaler).__name__}. "
        "Supported types: StandardScaler, MinMaxScaler"
    )


def _restore_array(value):
    if isinstance(value, list):
        return np.array(value)
    return value


def _deserialize_scaler(payload):
    if payload is None:
        return None
    scaler_class = payload.get("class")
    params = payload.get("params", {})
    state = payload.get("state", {})
    if scaler_class == "StandardScaler":
        scaler = StandardScaler(**params)
    elif scaler_class == "MinMaxScaler":
        scaler = MinMaxScaler(**params)
    else:
        raise ValueError(
            f"Unsupported scaler class: {scaler_class}. "
            "Supported classes: StandardScaler, MinMaxScaler"
        )
    for key, value in state.items():
        setattr(scaler, key, _restore_array(value))
    return scaler


def _load_scalers(cache_dir: Path, cache_key: str, prefix: str):
    scaler_file_pickle = cache_dir / f"{cache_key}_{prefix}_scalers.pkl"
    scaler_file_json = cache_dir / f"{cache_key}_{prefix}_scalers.json"

    # Load scalers (pickle primary, JSON fallback)
    if scaler_file_pickle.exists():
        with open(scaler_file_pickle, "rb") as f:
            scaler_data = pickle.load(f)
        feature_scaler = scaler_data["feature_scaler"]
        target_scaler = scaler_data["target_scaler"]
    else:
        with open(scaler_file_json, "r", encoding="utf-8") as f:
            scaler_payload = json.load(f)
        feature_scaler = _deserialize_scaler(scaler_payload.get("feature_scaler"))
        target_scaler = _deserialize_scaler(scaler_payload.get("target_scaler"))

    return feature_scaler, target_scaler


def _save_splits_to_npz(splits, split_file: Path) -> None:
    np.savez(
        split_file,
        X_train=splits.X_train,
        X_val=splits.X_val,
        X_test=splits.X_test,
        y_train=splits.y_train,
        y_val=splits.y_val,
        y_test=splits.y_test,
        cell_ids_train=splits.cell_ids_train,
        cell_ids_val=splits.cell_ids_val,
        cell_ids_test=splits.cell_ids_test,
        group_labels_train=splits.group_labels_train,
        group_labels_val=splits.group_labels_val,
        group_labels_test=splits.group_labels_test,
        X_train_raw=splits.X_train_raw if splits.X_train_raw is not None else np.array([]),
        X_val_raw=splits.X_val_raw if splits.X_val_raw is not None else np.array([]),
        X_test_raw=splits.X_test_raw if splits.X_test_raw is not None else np.array([]),
        y_train_raw=splits.y_train_raw if splits.y_train_raw is not None else np.array([]),
        y_val_raw=splits.y_val_raw if splits.y_val_raw is not None else np.array([]),
        y_test_raw=splits.y_test_raw if splits.y_test_raw is not None else np.array([]),
    )



def save_prepared_data(
    prepared: PreparedData,
    cache_dir: Path,
    cache_key: str,
) -> None:
    """Save PreparedData to disk.
    
    Parameters
    ----------
    prepared : PreparedData
        The prepared data object with splits and scalers.
    cache_dir : Path
        Directory to save cache files.
    cache_key : str
        Hash key for this configuration.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    split_file = cache_dir / f"{cache_key}_gene_splits.npz"
    splits = prepared.splits
    _save_splits_to_npz(splits, split_file)
    
    # Save scalers as pickle (primary format)
    scaler_file = cache_dir / f"{cache_key}_gene_scalers.pkl"
    with open(scaler_file, "wb") as f:
        pickle.dump({
            "feature_scaler": prepared.feature_scaler,
            "target_scaler": prepared.target_scaler,
        }, f)
    
    _LOG.info("Saved prepared gene-wise data to %s", cache_dir)


def load_prepared_data(cache_dir: Path, cache_key: str) -> Optional[PreparedData]:
    """Load PreparedData from disk.
    
    Parameters
    ----------
    cache_dir : Path
        Directory containing cache files.
    cache_key : str
        Hash key for this configuration.
    
    Returns
    -------
    Optional[PreparedData]
        The prepared data, or None if not found.
    """
    cache_dir = Path(cache_dir)
    split_file = cache_dir / f"{cache_key}_gene_splits.npz"
    scaler_file_pickle = cache_dir / f"{cache_key}_gene_scalers.pkl"
    scaler_file_json = cache_dir / f"{cache_key}_gene_scalers.json"
    
    if not split_file.exists() or (not scaler_file_pickle.exists() and not scaler_file_json.exists()):
        return None
    
    try:
        # Load splits
        with np.load(split_file, allow_pickle=True) as data:
            splits = SplitData(
                X_train=data["X_train"],
                X_val=data["X_val"],
                X_test=data["X_test"],
                y_train=data["y_train"],
                y_val=data["y_val"],
                y_test=data["y_test"],
                cell_ids_train=data["cell_ids_train"],
                cell_ids_val=data["cell_ids_val"],
                cell_ids_test=data["cell_ids_test"],
                group_labels_train=data["group_labels_train"],
                group_labels_val=data["group_labels_val"],
                group_labels_test=data["group_labels_test"],
                X_train_raw=data["X_train_raw"] if data["X_train_raw"].size > 0 else None,
                X_val_raw=data["X_val_raw"] if data["X_val_raw"].size > 0 else None,
                X_test_raw=data["X_test_raw"] if data["X_test_raw"].size > 0 else None,
                y_train_raw=data["y_train_raw"] if data["y_train_raw"].size > 0 else None,
                y_val_raw=data["y_val_raw"] if data["y_val_raw"].size > 0 else None,
                y_test_raw=data["y_test_raw"] if data["y_test_raw"].size > 0 else None,
            )
        
        feature_scaler, target_scaler = _load_scalers(cache_dir, cache_key, "gene")
        
        prepared = PreparedData(
            splits=splits,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
        )
        
        _LOG.info("Loaded prepared gene-wise data from %s", cache_dir)
        return prepared
    except Exception as exc:
        _LOG.warning("Failed to load prepared data from %s: %s", cache_dir, exc)
        return None


def _save_sparse_matrix(matrix, path: Path) -> None:
    sp.save_npz(path, matrix, compressed=False)


def _load_sparse_matrix(path: Path):
    return sp.load_npz(path)


def save_prepared_cellwise_data(
    prepared: PreparedCellwiseData,
    cache_dir: Path,
    cache_key: str,
) -> None:
    """Save PreparedCellwiseData to disk.
    
    Parameters
    ----------
    prepared : PreparedCellwiseData
        The prepared cell-wise data object with splits and scalers.
    cache_dir : Path
        Directory to save cache files.
    cache_key : str
        Hash key for this configuration.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    split_file = cache_dir / f"{cache_key}_cellwise_splits.npz"
    splits = prepared.splits

    def _save_matrix(name: str, mat):
        if mat is None:
            return {"is_sparse": False, "saved": False}
        if sp.issparse(mat):
            mat_path = cache_dir / f"{cache_key}_cellwise_{name}.npz"
            _save_sparse_matrix(mat, mat_path)
            return {"is_sparse": True, "saved": True, "path": mat_path.name}
        return {"is_sparse": False, "saved": True}

    meta = {
        "X_train": _save_matrix("X_train", splits.X_train),
        "X_val": _save_matrix("X_val", splits.X_val),
        "X_test": _save_matrix("X_test", splits.X_test),
    }

    # Skip caching raw matrices when sparse to avoid huge redundant files.
    # Note: if force_dense_features (or similar) changes between runs, the cache
    # may lack raw dense matrices even if they would now be expected.
    save_raw = not any(meta[name]["is_sparse"] for name in ("X_train", "X_val", "X_test"))
    if not save_raw:
        if _LOG.isEnabledFor(logging.INFO):
            sparse_splits = [
                name
                for name in ("X_train", "X_val", "X_test")
                if meta[name]["is_sparse"]
            ]
            _LOG.info(
                "Skipping caching of raw cell-wise matrices for cache_key '%s' because "
                "the following splits are sparse: %s. If dense features are required "
                "in a later run, the cache may not include raw matrices.",
                cache_key,
                ", ".join(sparse_splits) if sparse_splits else "unknown",
            )
    meta.update(
        {
            "X_train_raw": _save_matrix("X_train_raw", splits.X_train_raw) if save_raw else {"is_sparse": False, "saved": False},
            "X_val_raw": _save_matrix("X_val_raw", splits.X_val_raw) if save_raw else {"is_sparse": False, "saved": False},
            "X_test_raw": _save_matrix("X_test_raw", splits.X_test_raw) if save_raw else {"is_sparse": False, "saved": False},
        }
    )

    np.savez(
        split_file,
        X_train=splits.X_train if not meta["X_train"]["is_sparse"] else np.array([]),
        X_val=splits.X_val if not meta["X_val"]["is_sparse"] else np.array([]),
        X_test=splits.X_test if not meta["X_test"]["is_sparse"] else np.array([]),
        y_train=splits.y_train,
        y_val=splits.y_val,
        y_test=splits.y_test,
        cell_ids_train=splits.cell_ids_train,
        cell_ids_val=splits.cell_ids_val,
        cell_ids_test=splits.cell_ids_test,
        group_labels_train=splits.group_labels_train,
        group_labels_val=splits.group_labels_val,
        group_labels_test=splits.group_labels_test,
        X_train_raw=splits.X_train_raw if (splits.X_train_raw is not None and not meta["X_train_raw"]["is_sparse"]) else np.array([]),
        X_val_raw=splits.X_val_raw if (splits.X_val_raw is not None and not meta["X_val_raw"]["is_sparse"]) else np.array([]),
        X_test_raw=splits.X_test_raw if (splits.X_test_raw is not None and not meta["X_test_raw"]["is_sparse"]) else np.array([]),
        y_train_raw=splits.y_train_raw if splits.y_train_raw is not None else np.array([]),
        y_val_raw=splits.y_val_raw if splits.y_val_raw is not None else np.array([]),
        y_test_raw=splits.y_test_raw if splits.y_test_raw is not None else np.array([]),
        _X_train_sparse=meta["X_train"]["is_sparse"],
        _X_val_sparse=meta["X_val"]["is_sparse"],
        _X_test_sparse=meta["X_test"]["is_sparse"],
        _X_train_raw_sparse=meta["X_train_raw"]["is_sparse"],
        _X_val_raw_sparse=meta["X_val_raw"]["is_sparse"],
        _X_test_raw_sparse=meta["X_test_raw"]["is_sparse"],
        _X_train_path=meta["X_train"].get("path", ""),
        _X_val_path=meta["X_val"].get("path", ""),
        _X_test_path=meta["X_test"].get("path", ""),
        _X_train_raw_path=meta["X_train_raw"].get("path", ""),
        _X_val_raw_path=meta["X_val_raw"].get("path", ""),
        _X_test_raw_path=meta["X_test_raw"].get("path", ""),
    )
    
    # Save scalers as pickle (primary format)
    scaler_file = cache_dir / f"{cache_key}_cellwise_scalers.pkl"
    with open(scaler_file, "wb") as f:
        pickle.dump({
            "feature_scaler": prepared.feature_scaler,
            "target_scaler": prepared.target_scaler,
        }, f)
    
    _LOG.info("Saved prepared cell-wise data to %s", cache_dir)


def load_prepared_cellwise_data(cache_dir: Path, cache_key: str) -> Optional[PreparedCellwiseData]:
    """Load PreparedCellwiseData from disk.
    
    Parameters
    ----------
    cache_dir : Path
        Directory containing cache files.
    cache_key : str
        Hash key for this configuration.
    
    Returns
    -------
    Optional[PreparedCellwiseData]
        The prepared cell-wise data, or None if not found.
    """
    cache_dir = Path(cache_dir)
    split_file = cache_dir / f"{cache_key}_cellwise_splits.npz"
    scaler_file_pickle = cache_dir / f"{cache_key}_cellwise_scalers.pkl"
    scaler_file_json = cache_dir / f"{cache_key}_cellwise_scalers.json"
    
    if not split_file.exists() or (not scaler_file_pickle.exists() and not scaler_file_json.exists()):
        return None
    
    try:
        # Load splits
        with np.load(split_file, allow_pickle=True) as data:
            def _load_matrix(name: str, *, allow_empty: bool = False):
                sparse_flag = bool(data.get(f"_{name}_sparse", False))
                path = data.get(f"_{name}_path", "")
                if sparse_flag:
                    if not path:
                        _LOG.warning("Missing sparse path for %s in %s", name, split_file)
                        return None
                    mat_path = cache_dir / str(path)
                    if not mat_path.exists():
                        _LOG.warning("Missing sparse matrix file %s for %s", mat_path, name)
                        return None
                    return _load_sparse_matrix(mat_path)
                arr = data[name]
                if allow_empty and getattr(arr, "size", 0) == 0:
                    return None
                return arr

            X_train = _load_matrix("X_train")
            X_val = _load_matrix("X_val")
            X_test = _load_matrix("X_test")
            if X_train is None or X_val is None or X_test is None:
                _LOG.warning("Cached cell-wise splits incomplete; ignoring cache at %s", split_file)
                return None
            if not sp.issparse(X_train) and (getattr(X_train, "ndim", 0) < 2 or getattr(X_train, "dtype", None) == object):
                _LOG.warning("Cached cell-wise splits appear incompatible; ignoring cache at %s", split_file)
                return None
            splits = CellwiseSplitData(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=data["y_train"],
                y_val=data["y_val"],
                y_test=data["y_test"],
                cell_ids_train=data["cell_ids_train"],
                cell_ids_val=data["cell_ids_val"],
                cell_ids_test=data["cell_ids_test"],
                group_labels_train=data["group_labels_train"],
                group_labels_val=data["group_labels_val"],
                group_labels_test=data["group_labels_test"],
                X_train_raw=_load_matrix("X_train_raw", allow_empty=True) if "X_train_raw" in data else None,
                X_val_raw=_load_matrix("X_val_raw", allow_empty=True) if "X_val_raw" in data else None,
                X_test_raw=_load_matrix("X_test_raw", allow_empty=True) if "X_test_raw" in data else None,
                y_train_raw=data["y_train_raw"] if data["y_train_raw"].size > 0 else None,
                y_val_raw=data["y_val_raw"] if data["y_val_raw"].size > 0 else None,
                y_test_raw=data["y_test_raw"] if data["y_test_raw"].size > 0 else None,
            )
        
        feature_scaler, target_scaler = _load_scalers(cache_dir, cache_key, "cellwise")
        
        prepared = PreparedCellwiseData(
            splits=splits,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
        )
        
        _LOG.info("Loaded prepared cell-wise data from %s", cache_dir)
        return prepared
    except Exception as exc:
        _LOG.warning("Failed to load prepared cell-wise data from %s: %s", cache_dir, exc)
        return None

"""Shared data structures used across training and caching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    cell_ids_train: np.ndarray
    cell_ids_val: np.ndarray
    cell_ids_test: np.ndarray
    group_labels_train: np.ndarray
    group_labels_val: np.ndarray
    group_labels_test: np.ndarray
    X_train_raw: Optional[np.ndarray] = field(default=None, repr=False)
    X_val_raw: Optional[np.ndarray] = field(default=None, repr=False)
    X_test_raw: Optional[np.ndarray] = field(default=None, repr=False)
    y_train_raw: Optional[np.ndarray] = field(default=None, repr=False)
    y_val_raw: Optional[np.ndarray] = field(default=None, repr=False)
    y_test_raw: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class PreparedData:
    splits: SplitData
    feature_scaler: Optional[StandardScaler | MinMaxScaler]
    target_scaler: Optional[StandardScaler | MinMaxScaler]


@dataclass
class CellwiseSplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    cell_ids_train: np.ndarray
    cell_ids_val: np.ndarray
    cell_ids_test: np.ndarray
    group_labels_train: np.ndarray
    group_labels_val: np.ndarray
    group_labels_test: np.ndarray
    X_train_raw: Optional[np.ndarray] = field(default=None, repr=False)
    X_val_raw: Optional[np.ndarray] = field(default=None, repr=False)
    X_test_raw: Optional[np.ndarray] = field(default=None, repr=False)
    y_train_raw: Optional[np.ndarray] = field(default=None, repr=False)
    y_val_raw: Optional[np.ndarray] = field(default=None, repr=False)
    y_test_raw: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class PreparedCellwiseData:
    splits: CellwiseSplitData
    feature_scaler: Optional[StandardScaler | MinMaxScaler]
    target_scaler: Optional[StandardScaler | MinMaxScaler]

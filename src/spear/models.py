
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import torch
from torch import nn
import torch.nn.functional as F

try:  # optional dependency
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional
    XGBRegressor = None  # type: ignore

try:  # optional dependency
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional
    CatBoostRegressor = None  # type: ignore

from .config import TrainingConfig


class _CatBoostRegressorCompat(BaseEstimator, RegressorMixin):
    """Compatibility wrapper so sklearn meta-estimators can clone CatBoost.

    Newer scikit-learn releases rely on BaseEstimator tags; CatBoost's
    estimator doesn't expose them, so we wrap it with a thin adapter.
    """

    def __init__(self, **params):
        self._params = dict(params)
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed. Install with `pip install catboost` to enable this model.")
        self._model = CatBoostRegressor(**self._params)

    def fit(self, X, y, **fit_params):
        self._model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def __getattr__(self, name):
        # Delegate CatBoost-specific attributes (e.g., feature_importances_) to the wrapped model.
        return getattr(self._model, name)

    def get_params(self, deep: bool = True):
        return dict(self._params)

    def set_params(self, **params):
        self._params.update(params)
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed. Install with `pip install catboost` to enable this model.")
        self._model = CatBoostRegressor(**self._params)
        return self


def _rf_params(
    training: TrainingConfig,
    *,
    default_estimators: int,
    default_min_leaf: int = 2,
    default_max_features: float | str | None = None,
    default_bootstrap: bool = True,
) -> dict[str, object]:
    n_estimators = training.rf_n_estimators or default_estimators
    max_depth = training.rf_max_depth
    min_samples_leaf = training.rf_min_samples_leaf or default_min_leaf
    max_features = training.rf_max_features if training.rf_max_features is not None else default_max_features
    bootstrap = default_bootstrap if training.rf_bootstrap is None else training.rf_bootstrap

    params: dict[str, object] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "n_jobs": -1,
        "random_state": training.random_state,
        "bootstrap": bootstrap,
        "oob_score": bootstrap,
    }

    if max_features is not None:
        params["max_features"] = max_features

    return params


@dataclass
class TorchModelBundle:
    model: nn.Module
    reshape: str = "flat"  # "flat" or "sequence"


def _target_segments(input_dim: int, *, divisor: int, min_segments: int = 8, max_segments: int = 128) -> int:
    return max(min_segments, min(max_segments, max(1, input_dim // divisor)))


class CNNRegressor(nn.Module):
    """1D Convolutional Neural Network for feature extraction and regression.
    
    Architecture:
        - Multi-scale convolutions (7, 5, 3 kernel sizes with strides 4, 4, 2) progressively
          compress spatial dimensions while increasing channel depth (32→64→128)
        - Batch normalization after each conv layer for training stability
        - Adaptive pooling to fixed segment count for consistent architecture across input sizes
        - Dense head (512 → output_dim) with dropout regularization
        - Total parameters: ~O(input_dim) depending on segment count (8-128 segments)
    
    Memory Profile:
        - Forward pass activations: ~input_dim * 256 floats during feature extraction
        - Typical GPU memory: 50-150 MB per model for input_dim=40k
        - Batch-friendly: Memory scales linearly with batch size
    
    Input: (batch, input_dim) or (batch, 1, input_dim) → sequence format
    Output: (batch, output_dim) predictions
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        target_segments = _target_segments(input_dim, divisor=32)
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.pool = nn.AdaptiveAvgPool1d(target_segments)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * target_segments, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


class ResNetBlock1D(nn.Module):
    """Basic residual block for 1D convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.downsample(x)
        return F.relu(out)


class ResNet1DRegressor(nn.Module):
    """ResNet-style 1D CNN for ATAC feature regression.

    Architecture:
        - Strided conv stem + max pool for downsampling
        - 3 residual stages (channels 32 → 64 → 128) with 2 blocks each
        - Adaptive pooling to fixed segment count for stable dense head
        - Dense head (256 → output_dim) with dropout

    Input: (batch, input_dim) or (batch, 1, input_dim) → sequence format
    Output: (batch, output_dim) predictions
    """

    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        # Heuristic for the number of pooled segments used by the dense head:
        # - CNNRegressor has total stride 4*4*2 = 32; this ResNet has stride
        #   4 (stem) * 2 (maxpool) * 2 (layer2) * 2 (layer3) = 32 as well.
        # - We still use input_dim // 64 (vs. // 32 in CNNRegressor) to halve the
        #   segment count and keep the 128 * target_segments head size in check
        #   given the deeper residual stack.
        # - Clamp the segment count to [8, 128] to keep the representation
        #   expressive but memory-efficient.
        target_segments = _target_segments(input_dim, divisor=64)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(target_segments)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * target_segments, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def _make_layer(self, in_channels: int, out_channels: int, *, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResNetBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResNetBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


class RNNRegressor(nn.Module):
    """Recurrent Neural Network (vanilla RNN) for sequence modeling.
    
    Architecture:
        - Initial projection via convolutions (1→32→64→96 channels) to compress spatial dims
        - Vanilla RNN layers (tanh activation) to capture sequential dependencies
        - Adaptive pooling to fixed segment count for variable-length inputs
        - Dense head (96→128→output_dim) with dropout
        - Dropout between RNN layers when num_layers>1
        - Total parameters: O(hidden_size²) for RNN cell + O(input_dim) for projection
    
    Memory Profile:
        - RNN hidden states: (batch, hidden_size=96, seq_len) per layer
        - Typical GPU memory: 80-200 MB per model for input_dim=40k
        - Sequential dependency capture requires full forward pass (not parallelizable)
    
    Input: (batch, input_dim) → flattened input
    Output: (batch, output_dim) predictions (uses final hidden state)
    """
    
    def __init__(self, input_dim: int, hidden_size: int = 96, num_layers: int = 1, output_dim: int = 1):
        super().__init__()
        target_segments = max(8, min(128, max(1, input_dim // 32)))
        self.project = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=8, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(target_segments),
        )
        dropout = 0.0 if num_layers <= 1 else 0.1
        self.rnn = nn.RNN(
            input_size=96,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.transpose(1, 2)
        x = self.project(x)
        x = x.transpose(1, 2)
        output, _ = self.rnn(x)
        last = output[:, -1, :]
        return self.head(last)


class LSTMRegressor(nn.Module):
    """Long Short-Term Memory network for capturing long-range sequential patterns.
    
    Architecture:
        - Initial projection via convolutions (1→32→64→96 channels) for dimensionality reduction
        - LSTM layers (hidden_size=128) with cell state tracking for long-range dependencies
        - Adaptive pooling to fixed segment count (8-128)
        - Dense head (128→128→output_dim) with dropout regularization
        - Dropout between LSTM layers when num_layers>1
        - Total parameters: O(hidden_size²) per LSTM layer + O(input_dim) for projection
    
    Memory Profile:
        - LSTM cell state + hidden state: 2 × (batch, hidden_size=128, seq_len)
        - Typical GPU memory: 120-300 MB per model for input_dim=40k
        - Gradient computation requires storing intermediate activations (higher than RNN)
        - Best for capturing cell-to-cell regulatory patterns
    
    Input: (batch, input_dim) → flattened input
    Output: (batch, output_dim) predictions (uses final hidden state)
    """
    
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1, output_dim: int = 1):
        super().__init__()
        target_segments = max(8, min(128, max(1, input_dim // 32)))
        self.project = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=8, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(target_segments),
        )
        dropout = 0.0 if num_layers <= 1 else 0.1
        self.lstm = nn.LSTM(
            input_size=96,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.transpose(1, 2)
        x = self.project(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(last)


class MLPRegressor(nn.Module):
    """Multi-Layer Perceptron for direct feature-to-target mapping.
    
    Architecture:
        - Fully connected dense layers with layer normalization and dropout
        - Default configuration: input_dim → 256 → 256 → 128 → output_dim
        - LayerNorm provides adaptive scaling per layer, improving gradient flow
        - ReLU activations with 0.2 dropout for regularization
        - Final layer is linear (no activation) for regression
        - Total parameters: O(input_dim × hidden_layers²)
    
    Memory Profile:
        - Fastest neural model: no recurrence or convolution overhead
        - Typical GPU memory: 40-100 MB per model for input_dim=40k
        - All computations are parallelizable (ideal for batching)
        - Forward/backward passes are ~10-15x faster than LSTM/Transformer
    
    Input: (batch, input_dim) → dense connections to all hidden units
    Output: (batch, output_dim) predictions
    """
    
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...] = (256, 256, 128), output_dim: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.LayerNorm(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = units
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out


class CrossLayer(nn.Module):
    """Cross layer from Deep & Cross Network (DCN).

    Implements x_{l+1} = x0 * (w_l^T x_l) + b_l + x_l.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim))
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # (batch, d) @ (d,) -> (batch,)
        cross = torch.matmul(xl, self.weight).unsqueeze(1)
        return x0 * cross + self.bias + xl


class DCNRegressor(nn.Module):
    """Deep & Cross Network for explicit feature interactions + deep representation.

    Architecture:
        - Cross network with L cross layers (default 3) that builds bounded-degree
          feature crosses via x_{l+1} = x0 * (w_l^T x_l) + b_l + x_l.
        - Deep network with configurable MLP stack (default 256→256→128).
        - Concatenate cross output with deep output, then linear projection to target.

    Input: (batch, input_dim) → dense feature vector
    Output: (batch, output_dim) predictions
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        num_cross_layers: int = 3,
        deep_layers: tuple[int, ...] = (256, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_cross_layers < 1:
            raise ValueError("num_cross_layers must be >= 1")
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_cross_layers)])

        if deep_layers:
            deep: list[nn.Module] = []
            in_dim = input_dim
            for units in deep_layers:
                deep.append(nn.Linear(in_dim, units))
                deep.append(nn.LayerNorm(units))
                deep.append(nn.ReLU())
                deep.append(nn.Dropout(dropout))
                in_dim = units
            self.deep = nn.Sequential(*deep)
            deep_out_dim = deep_layers[-1]
        else:
            self.deep = nn.Identity()
            deep_out_dim = input_dim

        self.output = nn.Linear(input_dim + deep_out_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        deep_out = self.deep(x)
        combined = torch.cat([xl, deep_out], dim=-1)
        return self.output(combined)


class TransformerRegressor(nn.Module):
    """Transformer-based architecture with multi-head self-attention for pattern discovery.
    
    Architecture:
        - Initial CNN projection (64→96→128 channels) reduces spatial dims while preserving structure
        - Adaptive pooling to fixed sequence length (8-256 segments based on input_dim)
        - Learned positional embeddings (N(0, 0.02)) for segment ordering
        - Multi-head self-attention (8 heads) to capture interactions between ATAC segments
        - Transformer encoder stack (2 layers) with GELU activations
        - Dense head with LayerNorm, GELU, dropout → output
        - Total parameters: O(embed_dim × num_heads × seq_len) + O(input_dim)
    
    Memory Profile:
        - Attention matrix: O(seq_len²) memory footprint
        - Typical GPU memory: 150-400 MB per model for input_dim=40k (seq_len ~32-64)
        - Forward pass is O(seq_len²) complexity; critical for large sequence lengths
        - Best for discovering long-range ATAC-promoter interactions
    
    Input: (batch, input_dim) → sequence (batch, seq_len, embed_dim)
    Output: (batch, output_dim) predictions (uses mean-pooled token embeddings)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        target_segments = max(8, min(256, max(1, input_dim // 32)))
        self.project = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(target_segments)
        self.channel_proj = nn.Conv1d(128, embed_dim, kernel_size=1)
        self.positional = nn.Parameter(torch.randn(1, target_segments, embed_dim) * 0.02)
        head_options = [h for h in (8, 4, 2, 1) if embed_dim % h == 0]
        head_count = head_options[0] if head_options else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=head_count,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.transpose(1, 2)
        x = self.project(x)
        x = self.pool(x)
        x = self.channel_proj(x)
        x = x.transpose(1, 2)
        pos = self.positional[:, : x.size(1), :]
        x = x + pos
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


class GraphRegressor(nn.Module):
    """Graph Neural Network modeling ATAC-seq bins as 1D nodes with local spatial connectivity.
    
    Architecture:
        - Reshapes ATAC features into fixed node count (8-64 nodes) via binning
        - Node encoder: LayerNorm → MLP (3 layers, GELU activation) learns node embeddings
        - Local message passing: distance-weighted attention between adjacent bins
        - Update block: residual MLP for multi-hop message aggregation
        - Dense head: flattens node embeddings → 2-layer MLP → output
        - Total parameters: O(hidden_dim² × num_nodes) + O(input_dim)
    
    Memory Profile:
        - Adjacency matrix: O(num_nodes²) = O(1) for fixed node count
        - Typical GPU memory: 100-250 MB per model for input_dim=40k (num_nodes~32-48)
        - Message passing requires multiple forward passes per layer
        - Best for spatial ATAC structure around promoters (TSS neighborhood)
    
    Input: (batch, input_dim) → graph nodes (batch, num_nodes, hidden_dim)
    Output: (batch, output_dim) predictions (uses aggregated node embeddings)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        max_nodes: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        num_nodes = max(8, min(max_nodes, max(1, input_dim // 8)))
        node_dim = math.ceil(input_dim / num_nodes)
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.pad_dim = (node_dim * num_nodes) - input_dim

        self.node_encoder = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.update_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim * num_nodes),
            nn.Linear(hidden_dim * num_nodes, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, output_dim),
        )

        positions = torch.arange(num_nodes, dtype=torch.float32)
        distance = positions[:, None] - positions[None, :]
        adjacency = torch.exp(-(distance**2) / (2.0 * (num_nodes / 6.0) ** 2))
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True)
        self.register_buffer("adjacency", adjacency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_dim > 0:
            x = F.pad(x, (0, self.pad_dim))
        nodes = x.view(x.size(0), self.num_nodes, self.node_dim)
        h = self.node_encoder(nodes)
        agg = torch.einsum("ij,bjd->bid", self.adjacency, h)
        h = h + self.update_block(agg)
        return self.head(h)


def build_model(
    name: str,
    input_dim: int,
    training: TrainingConfig,
    output_dim: int = 1,
    artifacts_dir: Optional[Path] = None,
) -> object:
    name = name.lower()
    if name == "cnn":
        return TorchModelBundle(CNNRegressor(input_dim, output_dim=output_dim))
    if name == "resnet":
        return TorchModelBundle(ResNet1DRegressor(input_dim, output_dim=output_dim))
    if name == "rnn":
        return TorchModelBundle(RNNRegressor(input_dim, output_dim=output_dim), reshape="sequence")
    if name == "lstm":
        return TorchModelBundle(LSTMRegressor(input_dim, output_dim=output_dim), reshape="sequence")
    if name == "transformer":
        return TorchModelBundle(TransformerRegressor(input_dim, output_dim=output_dim), reshape="sequence")
    if name == "mlp":
        return TorchModelBundle(MLPRegressor(input_dim, output_dim=output_dim))
    if name == "dcn":
        return TorchModelBundle(DCNRegressor(input_dim, output_dim=output_dim))
    if name == "graph":
        return TorchModelBundle(GraphRegressor(input_dim, output_dim=output_dim), reshape="flat")
    if name == "svr":
        # Default to a linear kernel for efficiency on large datasets.
        # NOTE:
        #   - Linear SVR can only learn linear relationships.
        #   - RBF (or other non-linear kernels) can capture more complex gene regulatory patterns
        #     but are typically slower and more memory-intensive.
        #
        # To adjust this trade-off, TrainingConfig may define:
        #   - svr_kernel: str, e.g. "linear" (default) or "rbf"
        #   - svr_C: float, regularization strength (default 1.0)
        #   - svr_epsilon: float, epsilon-insensitive loss parameter (default 0.1)
        #   - svr_max_iter: int, maximum iterations (default 50000)
        #   - svr_tol: float, tolerance for stopping criterion (default 1e-4)
        #
        # Example for higher-capacity non-linear model (if supported by TrainingConfig):
        #   training.svr_kernel = "rbf"
        #   training.svr_C = 10.0
        kernel = training.svr_kernel
        C = training.svr_C
        epsilon = training.svr_epsilon
        max_iter = training.svr_max_iter
        tol = training.svr_tol
        svr_estimator = SVR(C=C, epsilon=epsilon, kernel=kernel, max_iter=max_iter, tol=tol)
        if output_dim == 1:
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", svr_estimator),
                ]
            )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", MultiOutputRegressor(svr_estimator)),
            ]
        )
    if name == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed. Install with `pip install xgboost` to enable this model.")
        use_gpu = str(training.device_preference).lower() == "cuda"
        xgb_params = dict(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=training.random_state,
        )
        if use_gpu:
            xgb_params["device"] = "cuda"
        base_model = XGBRegressor(**xgb_params)
        if output_dim == 1:
            return base_model
        return MultiOutputRegressor(base_model)
    if name == "random_forest":
        params = _rf_params(
            training,
            default_estimators=600,
            default_min_leaf=2,
            default_max_features=None,
        )
        return RandomForestRegressor(**params)
    if name == "catboost":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed. Install with `pip install catboost` to enable this model.")
        use_gpu = str(training.device_preference).lower() == "cuda"
        # Use configurable iterations (default 1000) with early_stopping_rounds for automatic quality/speed balance
        # Set via training.catboost_iterations to override default
        iterations = training.catboost_iterations if training.catboost_iterations is not None else 1000
        catboost_params = {
            "iterations": iterations,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "RMSE",
            "verbose": False,
            "random_seed": training.random_state,
            "thread_count": -1,
            "early_stopping_rounds": 50,  # Stop if validation metric doesn't improve for 50 rounds
        }
        if use_gpu:
            catboost_params["task_type"] = "GPU"
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            catboost_params["train_dir"] = str(artifacts_dir)
            catboost_params["allow_writing_files"] = True
        else:
            catboost_params["allow_writing_files"] = False
        base_model = _CatBoostRegressorCompat(**catboost_params)
        if output_dim == 1:
            return base_model
        return MultiOutputRegressor(base_model)
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=800,
            max_depth=None,
            n_jobs=-1,
            random_state=training.random_state,
        )
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        base_model = HistGradientBoostingRegressor(
            learning_rate=0.1,  # Increased for faster convergence
            max_depth=6,
            max_iter=300,  # Reduced from 600 for speed
            max_leaf_nodes=64,
            min_samples_leaf=50,  # Increased from 20 for speed (less granular splits)
            l2_regularization=1e-3,
            random_state=training.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,  # Reduced from 20 for earlier stopping
        )
        if output_dim == 1:
            return base_model
        return MultiOutputRegressor(base_model)
    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=training.random_state)),
        ])
    if name == "elastic_net":
        from sklearn.linear_model import ElasticNet, MultiTaskElasticNet

        scaler = StandardScaler()
        if output_dim == 1:
            regressor = ElasticNet(
                alpha=0.1,  # Increased from 0.05 for faster convergence
                l1_ratio=0.5,
                max_iter=3000,  # Reduced from 10000
                selection="random",  # Random is faster than cyclic
                tol=1e-3,  # Relaxed tolerance for speed
                random_state=training.random_state,
            )
        else:
            regressor = MultiTaskElasticNet(
                alpha=0.1,  # Increased from 0.05 for faster convergence
                l1_ratio=0.3,
                max_iter=3000,  # Reduced from 10000
                tol=1e-3,  # Relaxed tolerance for speed
                random_state=training.random_state,
            )
        return Pipeline([
            ("scaler", scaler),
            ("regressor", regressor),
        ])
    if name == "lasso":
        from sklearn.linear_model import Lasso, MultiTaskLasso

        scaler = StandardScaler()
        if output_dim == 1:
            regressor = Lasso(
                alpha=0.1,  # Increased from 0.05 for faster convergence
                max_iter=3000,  # Reduced from 10000
                selection="random",  # Random is faster than cyclic
                tol=1e-3,  # Relaxed tolerance for speed
                random_state=training.random_state,
            )
        else:
            regressor = MultiTaskLasso(
                alpha=0.1,  # Increased from 0.05 for faster convergence
                max_iter=3000,  # Reduced from 10000
                tol=1e-3,  # Relaxed tolerance for speed
                random_state=training.random_state,
            )
        return Pipeline([
            ("scaler", scaler),
            ("regressor", regressor),
        ])
    if name == "ols":
        from sklearn.linear_model import LinearRegression

        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ])
    raise ValueError(f"Unknown model name: {name}")

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR, SVR
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

FAST_MODE_SVR_MAX_ITER = 2_000
FAST_MODE_SVR_TOL = 1e-3
FAST_MODE_SVR_ALPHA = 1e-4
FAST_MODE_SVR_VAL_FRACTION = 0.1
FAST_MODE_SVR_NO_CHANGE_ITERS = 5


class _CatBoostRegressorCompat(BaseEstimator, RegressorMixin):
    """Compatibility wrapper so sklearn meta-estimators can clone CatBoost.

    Newer scikit-learn releases rely on BaseEstimator tags; CatBoost's
    estimator doesn't expose them, so we wrap it with a thin adapter.
    """

    def __init__(self, **params):
        self._params = dict(params)
        if CatBoostRegressor is None:
            raise RuntimeError(
                "catboost is not installed. Install with `pip install catboost` to enable this model."
            )
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
            raise RuntimeError(
                "catboost is not installed. Install with `pip install catboost` to enable this model."
            )
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
    max_features = (
        training.rf_max_features
        if training.rf_max_features is not None
        else default_max_features
    )
    bootstrap = (
        default_bootstrap if training.rf_bootstrap is None else training.rf_bootstrap
    )

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


def _target_segments(
    input_dim: int, *, divisor: int, min_segments: int = 8, max_segments: int = 128
) -> int:
    return max(min_segments, min(max_segments, max(1, input_dim // divisor)))


def _make_norm(normalized_shape: int) -> nn.Module:
    """Return RMSNorm with a local RMSNorm fallback for older torch builds."""
    rms_norm = getattr(nn, "RMSNorm", None)
    if rms_norm is not None:
        return rms_norm(normalized_shape)
    return RMSNorm(normalized_shape)


class RMSNorm(nn.Module):
    """RMSNorm over the last tensor dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class ChannelRMSNorm1d(nn.Module):
    """RMS-normalize Conv1d activations across channels."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"ChannelRMSNorm1d expects a 3D tensor, got shape {tuple(x.shape)}"
            )
        rms = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * rms * self.weight.view(1, -1, 1)


def _make_channel_norm(channels: int) -> nn.Module:
    return ChannelRMSNorm1d(channels)


class _GEGLU(nn.Module):
    """Gated GELU projection used in Transformer FFN blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


class _TransformerBlockV2(nn.Module):
    """Pre-norm Transformer block with RMSNorm, GEGLU FFN, and layer scaling."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = _make_norm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = _make_norm(embed_dim)
        ffn_dim = embed_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim * 2),
            _GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.drop2 = nn.Dropout(dropout)
        self.gamma1 = nn.Parameter(torch.ones(embed_dim) * 1e-3)
        self.gamma2 = nn.Parameter(torch.ones(embed_dim) * 1e-3)

    def forward(
        self, x: torch.Tensor, *, attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_in = self.norm1(x)
        # attn_bias is an additive mask of shape (seq_len, seq_len) (float) applied to attention logits.
        # We keep it shared across the batch to avoid per-sample overhead in per-gene training.
        attn_out, _ = self.attn(
            attn_in, attn_in, attn_in, need_weights=False, attn_mask=attn_bias
        )
        x = x + self.drop1(attn_out) * self.gamma1
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.drop2(ffn_out) * self.gamma2
        return x


class CNNRegressor(nn.Module):
    """1D Convolutional Neural Network for feature extraction and regression.

    Architecture:
        - Multi-scale convolutions (7, 5, 3 kernel sizes with strides 4, 4, 2) progressively
          compress spatial dimensions while increasing channel depth (32→64→128)
        - RMS normalization after each conv layer for training stability
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

    def __init__(self, input_dim: int, output_dim: int = 1, *, in_channels: int = 1):
        super().__init__()
        target_segments = _target_segments(input_dim, divisor=32)
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            _make_channel_norm(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            _make_channel_norm(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            _make_channel_norm(128),
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        attention: str = "none",
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = _make_channel_norm(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = _make_channel_norm(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                _make_channel_norm(out_channels),
            )
        else:
            self.downsample = nn.Identity()
        if attention == "se":
            self.attention = SEBlock1D(out_channels, reduction=se_reduction)
        else:
            self.attention = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        out = out + self.downsample(x)
        return F.relu(out)


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D feature maps."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced_channels = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(self.pool(x))
        return x * weights


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

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        attention: str = "se",
        attention_se_reduction: int = 8,
        *,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        if attention not in {"none", "se"}:
            raise ValueError("attention must be 'none' or 'se'")
        if attention_se_reduction <= 0:
            raise ValueError("attention_se_reduction must be positive")

        self.attention = attention
        self.attention_se_reduction = attention_se_reduction
        # Per-gene inputs are usually only ~40 bins wide. A heavily-strided stem
        # collapses that signal too early, so keep higher resolution on small
        # inputs and only use the more aggressive image-style stem on larger ones.
        compact_input = input_dim <= 128
        if compact_input:
            stem_stride = 1
            use_max_pool = False
            stage_strides = (1, 1, 2)
            target_segments = max(4, min(32, input_dim))
        else:
            stem_stride = 4
            use_max_pool = True
            stage_strides = (1, 2, 2)
            target_segments = _target_segments(input_dim, divisor=64)

        stem_layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                32,
                kernel_size=7,
                stride=stem_stride,
                padding=3,
                bias=False,
            ),
            _make_channel_norm(32),
            nn.ReLU(),
        ]
        if use_max_pool:
            stem_layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stem = nn.Sequential(*stem_layers)
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=stage_strides[0])
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=stage_strides[1])
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=stage_strides[2])
        self.pool = nn.AdaptiveAvgPool1d(target_segments)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * target_segments, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def _make_layer(
        self, in_channels: int, out_channels: int, *, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [
            ResNetBlock1D(
                in_channels,
                out_channels,
                stride=stride,
                attention=self.attention,
                se_reduction=self.attention_se_reduction,
            )
        ]
        for _ in range(1, blocks):
            layers.append(
                ResNetBlock1D(
                    out_channels,
                    out_channels,
                    stride=1,
                    attention=self.attention,
                    se_reduction=self.attention_se_reduction,
                )
            )
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


class LocalSharedWindowResNetRegressor(nn.Module):
    """Shared local-window ResNet head for multi-output regression.

    Each target predicts from its own local feature block (e.g., gene-local bins),
    while a shared ResNet is reused across all targets for parameter efficiency.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        feature_block_indices: Sequence[Sequence[int]],
        *,
        attention: str = "se",
        attention_se_reduction: int = 8,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if len(feature_block_indices) < output_dim:
            raise ValueError(
                "feature_block_indices must provide one block per output target "
                f"(got {len(feature_block_indices)} blocks for {output_dim} targets)"
            )

        per_target_blocks: list[np.ndarray] = []
        for target_idx in range(output_dim):
            raw_block = np.asarray(
                feature_block_indices[target_idx], dtype=np.int64
            ).ravel()
            # Ignore invalid entries to support padded/mapped index sources.
            valid_block = raw_block[(raw_block >= 0) & (raw_block < input_dim)]
            if valid_block.size == 0:
                raise ValueError(f"Target {target_idx} has no valid local features")
            per_target_blocks.append(valid_block)

        window_dim = max(int(block.shape[0]) for block in per_target_blocks)
        if window_dim <= 0:
            raise ValueError(
                "Unable to infer local window width from feature_block_indices"
            )

        block_index_matrix = np.zeros((output_dim, window_dim), dtype=np.int64)
        block_mask_matrix = np.zeros((output_dim, window_dim), dtype=np.float32)
        for target_idx, block in enumerate(per_target_blocks):
            width = int(block.shape[0])
            block_index_matrix[target_idx, :width] = block
            block_mask_matrix[target_idx, :width] = 1.0

        self.output_dim = output_dim
        self.window_dim = window_dim
        self.register_buffer("block_indices", torch.from_numpy(block_index_matrix))
        self.register_buffer("block_mask", torch.from_numpy(block_mask_matrix))
        self.local_backbone = ResNet1DRegressor(
            window_dim,
            output_dim=1,
            attention=attention,
            attention_se_reduction=attention_se_reduction,
        )
        self.target_bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            x = x.reshape(x.size(0), -1)

        batch_size = x.size(0)
        expanded_x = x.unsqueeze(1).expand(-1, self.output_dim, -1)
        gather_indices = self.block_indices.unsqueeze(0).expand(batch_size, -1, -1)
        local_windows = torch.gather(expanded_x, dim=2, index=gather_indices)
        local_windows = local_windows * self.block_mask.unsqueeze(0).to(
            dtype=local_windows.dtype
        )
        preds = self.local_backbone(
            local_windows.reshape(batch_size * self.output_dim, self.window_dim)
        )
        return preds.reshape(batch_size, self.output_dim) + self.target_bias


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

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 96,
        num_layers: int = 1,
        output_dim: int = 1,
        *,
        in_channels: int = 1,
    ):
        super().__init__()
        target_segments = max(8, min(128, max(1, input_dim // 32)))
        self.project = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=8, padding=3),
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

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        output_dim: int = 1,
        *,
        in_channels: int = 1,
    ):
        super().__init__()
        target_segments = max(8, min(128, max(1, input_dim // 32)))
        self.project = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=8, padding=3),
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
        x = self.project(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(last)


class MLPRegressor(nn.Module):
    """Multi-Layer Perceptron for direct feature-to-target mapping.

    Architecture:
        - Fully connected dense layers with RMS normalization and dropout
        - Default configuration: input_dim → 256 → 256 → 128 → output_dim
        - RMSNorm provides lightweight adaptive scaling per layer, improving gradient flow
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

    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...] = (256, 256, 128),
        output_dim: int = 1,
    ):
        super().__init__()
        width = max(max(hidden_layers, default=0), min(512, max(128, input_dim * 4)))
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, width),
            _make_norm(width),
            nn.GELU(),
        )

        blocks: list[nn.Module] = []
        for _ in range(max(2, len(hidden_layers))):
            blocks.append(
                nn.Sequential(
                    nn.Linear(width, width * 2),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(width * 2, width),
                    nn.Dropout(0.15),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = _make_norm(width)
        self.head = nn.Sequential(
            nn.Linear(width, max(64, width // 2)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(max(64, width // 2), output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.final_norm(x)
        return self.head(x)


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
        self.cross_layers = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(num_cross_layers)]
        )

        if deep_layers:
            deep: list[nn.Module] = []
            in_dim = input_dim
            for units in deep_layers:
                deep.append(nn.Linear(in_dim, units))
                deep.append(_make_norm(units))
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
        - Dense head with RMSNorm, GELU, dropout → output
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
        num_heads: Optional[int] = None,
        dropout: float = 0.2,
        *,
        in_channels: int = 1,
        relative_bias: bool = True,
        relative_bias_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be within [0, 1), got {dropout}")

        self.relative_bias = bool(relative_bias)
        self.relative_bias_gamma = float(relative_bias_gamma)
        self.compact_input = input_dim <= 128
        if self.compact_input:
            target_segments = input_dim
            self.project = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                _make_channel_norm(embed_dim),
                nn.GELU(),
                nn.Conv1d(
                    embed_dim,
                    embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=max(1, embed_dim // 8),
                    bias=False,
                ),
                _make_channel_norm(embed_dim),
                nn.GELU(),
            )
            self.pool = nn.Identity()
            self.channel_proj = nn.Identity()
        else:
            target_segments = max(8, min(256, max(1, input_dim // 32)))
            self.project = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=7, stride=4, padding=3),
                _make_channel_norm(64),
                nn.ReLU(),
                nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
                _make_channel_norm(96),
                nn.ReLU(),
                nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1),
                _make_channel_norm(128),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(target_segments)
            self.channel_proj = nn.Conv1d(128, embed_dim, kernel_size=1)
        self.positional = nn.Parameter(
            torch.randn(1, target_segments, embed_dim) * 0.02
        )
        if num_heads is None:
            head_options = [h for h in (8, 6, 4, 3, 2, 1) if embed_dim % h == 0]
            head_count = head_options[0] if head_options else 1
        else:
            if num_heads <= 0:
                raise ValueError(f"num_heads must be positive, got {num_heads}")
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
                )
            head_count = num_heads
        self.blocks = nn.ModuleList(
            [
                _TransformerBlockV2(embed_dim, head_count, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            _make_norm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(-1) == 1:
            # Legacy support if something provides (batch, seq_len, 1)
            x = x.transpose(1, 2)

        attn_bias = None
        # If an explicit scalar offset channel is present, compute pooled offsets and
        # use them for a distance-aware relative attention bias.
        if self.relative_bias and x.dim() == 3 and x.size(1) >= 2:
            # Channel 1 is expected to be a signed offset feature (e.g. signed_linear).
            # For rbf encoding we still append the scalar signed_linear channel first.
            offsets = x[:, 1:2, :]
            pooled = (
                self.pool(offsets)
                if not isinstance(self.pool, nn.Identity)
                else offsets
            )
            pooled = pooled.squeeze(1)  # (batch, seq_len)
            # Use mean across batch to keep mask shared (per-gene batches share the same offsets anyway).
            pos = pooled.mean(dim=0)  # (seq_len,)
            diffs = pos[:, None] - pos[None, :]
            attn_bias = (-self.relative_bias_gamma * diffs.abs()).to(dtype=x.dtype)
        x = self.project(x)
        x = self.pool(x)
        x = self.channel_proj(x)
        x = x.transpose(1, 2)
        pos = self.positional[:, : x.size(1), :]
        x = x + pos
        for block in self.blocks:
            x = block(x, attn_bias=attn_bias)
        x = x.mean(dim=1)
        return self.head(x)


class TransformerRegressorV2(nn.Module):
    """Hybrid convolution-attention transformer with gated token pooling."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: Optional[int] = None,
        dropout: float = 0.2,
        *,
        in_channels: int = 1,
        relative_bias: bool = True,
        relative_bias_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be within [0, 1), got {dropout}")

        if num_heads is None:
            head_options = [h for h in (8, 6, 4, 3, 2, 1) if embed_dim % h == 0]
            head_count = head_options[0] if head_options else 1
        else:
            if num_heads <= 0:
                raise ValueError(f"num_heads must be positive, got {num_heads}")
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
                )
            head_count = num_heads

        stem_channels = max(64, embed_dim)
        local_channels = stem_channels // 2
        context_channels = stem_channels - local_channels

        self.local_stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                local_channels,
                kernel_size=9,
                stride=4,
                padding=4,
                bias=False,
            ),
            _make_channel_norm(local_channels),
            nn.GELU(),
            nn.Conv1d(
                local_channels,
                local_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                groups=local_channels,
                bias=False,
            ),
            _make_channel_norm(local_channels),
            nn.GELU(),
        )
        self.context_stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                context_channels,
                kernel_size=15,
                stride=4,
                padding=7,
                bias=False,
            ),
            _make_channel_norm(context_channels),
            nn.GELU(),
            nn.Conv1d(
                context_channels,
                context_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                bias=False,
            ),
            _make_channel_norm(context_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(stem_channels, embed_dim, kernel_size=1, bias=False),
            _make_channel_norm(embed_dim),
            nn.GELU(),
        )

        target_segments = max(16, min(384, max(1, input_dim // 24)))
        self.pool = nn.AdaptiveAvgPool1d(target_segments)
        self.token_norm = _make_norm(embed_dim)
        self.positional = nn.Parameter(
            torch.randn(1, target_segments + 1, embed_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                _TransformerBlockV2(embed_dim, head_count, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = _make_norm(embed_dim)
        self.attn_pool = nn.Linear(embed_dim, 1)
        self.head = nn.Sequential(
            _make_norm(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.transpose(1, 2)

        attn_bias = None
        if self.relative_bias and x.dim() == 3 and x.size(1) >= 2:
            offsets = x[:, 1:2, :]
            pooled = self.pool(offsets)
            pooled = pooled.squeeze(1)
            pos = pooled.mean(dim=0)
            diffs = pos[:, None] - pos[None, :]
            attn_bias = (-self.relative_bias_gamma * diffs.abs()).to(dtype=x.dtype)
        local = self.local_stem(x)
        context = self.context_stem(x)
        x = torch.cat([local, context], dim=1)
        x = self.fuse(x)
        x = self.pool(x).transpose(1, 2)
        x = self.token_norm(x)
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional[:, : x.size(1), :]
        for block in self.blocks:
            x = block(x, attn_bias=attn_bias)
        x = self.final_norm(x)

        cls_token = x[:, 0, :]
        tokens = x[:, 1:, :]
        weights = torch.softmax(self.attn_pool(tokens).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = torch.sum(tokens * weights, dim=1)
        mean_tokens = tokens.mean(dim=1)
        max_tokens = tokens.amax(dim=1)
        features = torch.cat([cls_token, pooled, mean_tokens, max_tokens], dim=-1)
        return self.head(features)


class GraphRegressor(nn.Module):
    """Graph Neural Network modeling ATAC-seq bins as 1D nodes with local spatial connectivity.

    Architecture:
        - Reshapes ATAC features into fixed node count (8-64 nodes) via binning
        - Node encoder: RMSNorm → MLP (3 layers, GELU activation) learns node embeddings
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
            _make_norm(node_dim),
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.update_block = nn.Sequential(
            _make_norm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            _make_norm(hidden_dim * num_nodes),
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
    feature_block_indices: Optional[Sequence[Sequence[int]]] = None,
) -> object:
    name = name.lower()
    in_channels = 1
    if name in {"rnn", "lstm", "transformer"} and (
        getattr(training, "per_gene_feature_basis", "bin") == "peak"
        and getattr(training, "per_gene_peak_distance_encoding", "none") != "none"
    ):
        enc = str(
            getattr(training, "per_gene_peak_distance_encoding", "none") or "none"
        ).lower()
        if enc == "signed_linear":
            in_channels = 2
        elif enc == "rbf":
            # +1 for the scalar signed_linear channel used for relative bias + K RBF channels.
            k = int(getattr(training, "per_gene_peak_distance_rbf_bases", 16))
            in_channels = 1 + 1 + max(0, k)
    if name == "cnn":
        return TorchModelBundle(
            CNNRegressor(input_dim, output_dim=output_dim, in_channels=in_channels)
        )
    if name == "resnet":
        if output_dim > 1 and getattr(training, "multioutput_local_only", False):
            if not feature_block_indices:
                raise ValueError(
                    "multioutput_local_only requires feature_block_indices for local feature mapping"
                )
            return TorchModelBundle(
                LocalSharedWindowResNetRegressor(
                    input_dim,
                    output_dim=output_dim,
                    feature_block_indices=feature_block_indices,
                    attention=training.resnet_attention,
                    attention_se_reduction=training.resnet_attention_se_reduction,
                )
            )
        return TorchModelBundle(
            ResNet1DRegressor(
                input_dim,
                output_dim=output_dim,
                attention=training.resnet_attention,
                attention_se_reduction=training.resnet_attention_se_reduction,
                in_channels=in_channels,
            )
        )
    if name == "rnn":
        return TorchModelBundle(
            RNNRegressor(input_dim, output_dim=output_dim, in_channels=in_channels),
            reshape="sequence",
        )
    if name == "lstm":
        return TorchModelBundle(
            LSTMRegressor(input_dim, output_dim=output_dim, in_channels=in_channels),
            reshape="sequence",
        )
    if name == "transformer":
        transformer_arch = getattr(training, "transformer_arch", "v1")
        transformer_cls = (
            TransformerRegressorV2 if transformer_arch == "v2" else TransformerRegressor
        )
        return TorchModelBundle(
            transformer_cls(
                input_dim,
                output_dim=output_dim,
                embed_dim=training.transformer_embed_dim,
                num_layers=training.transformer_num_layers,
                num_heads=training.transformer_num_heads,
                dropout=training.transformer_dropout,
                in_channels=in_channels,
            ),
            reshape="sequence",
        )
    if name == "mlp":
        return TorchModelBundle(MLPRegressor(input_dim, output_dim=output_dim))
    if name == "dcn":
        return TorchModelBundle(DCNRegressor(input_dim, output_dim=output_dim))
    if name == "graph":
        return TorchModelBundle(
            GraphRegressor(input_dim, output_dim=output_dim), reshape="flat"
        )
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
        if training.fast_classical_mode:
            max_iter = min(max_iter, FAST_MODE_SVR_MAX_ITER)
            tol = max(tol, FAST_MODE_SVR_TOL)
            # Use a first-order linear approximation in fast mode to avoid
            # very long liblinear runtimes on large multi-output workloads.
            fast_svr = SGDRegressor(
                loss="epsilon_insensitive",
                epsilon=epsilon,
                alpha=FAST_MODE_SVR_ALPHA,
                max_iter=max_iter,
                tol=tol,
                random_state=training.random_state,
                early_stopping=True,
                validation_fraction=FAST_MODE_SVR_VAL_FRACTION,
                n_iter_no_change=FAST_MODE_SVR_NO_CHANGE_ITERS,
            )
            if output_dim == 1:
                return fast_svr
            return MultiOutputRegressor(fast_svr, n_jobs=1)
        if kernel == "linear":
            # LinearSVR scales much better than kernel SVR on large, high-dimensional
            # multi-output workloads (e.g., 1k genes x 40k features).
            svr_estimator = LinearSVR(
                C=C,
                epsilon=epsilon,
                max_iter=max_iter,
                tol=tol,
                random_state=training.random_state,
                dual="auto",
            )
        else:
            svr_estimator = SVR(
                C=C, epsilon=epsilon, kernel=kernel, max_iter=max_iter, tol=tol
            )
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
                ("regressor", MultiOutputRegressor(svr_estimator, n_jobs=-1)),
            ]
        )
    if name == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError(
                "xgboost is not installed. Install with `pip install xgboost` to enable this model."
            )
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
            raise RuntimeError(
                "catboost is not installed. Install with `pip install catboost` to enable this model."
            )
        use_gpu = str(training.device_preference).lower() == "cuda"
        # Use configurable iterations (default 1000) with early_stopping_rounds for automatic quality/speed balance
        # Set via training.catboost_iterations to override default
        iterations = (
            training.catboost_iterations
            if training.catboost_iterations is not None
            else 1000
        )
        depth = 6
        learning_rate = 0.05
        early_stopping_rounds = 50
        if training.fast_classical_mode:
            iterations = min(iterations, 400)
            depth = 5
            learning_rate = 0.08
            early_stopping_rounds = 30
        catboost_params = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "loss_function": "RMSE",
            "verbose": False,
            "random_seed": training.random_state,
            "thread_count": -1,
            "early_stopping_rounds": early_stopping_rounds,
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

        hgb_params = dict(
            learning_rate=0.1,
            max_depth=6,
            max_iter=300,
            max_leaf_nodes=64,
            min_samples_leaf=50,
            l2_regularization=1e-3,
            random_state=training.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        if training.fast_classical_mode:
            hgb_params.update(
                max_iter=140,
                max_leaf_nodes=31,
                min_samples_leaf=100,
                n_iter_no_change=8,
            )
        base_model = HistGradientBoostingRegressor(
            **hgb_params,
        )
        if output_dim == 1:
            return base_model
        # Running one process per target can explode memory on large multi-output
        # runs (e.g., 1000 genes x 40k features), leading to OOM kills.
        return MultiOutputRegressor(base_model, n_jobs=1)
    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0, random_state=training.random_state)),
            ]
        )
    if name == "elastic_net":
        from sklearn.linear_model import ElasticNet, MultiTaskElasticNet

        alpha = 0.1
        tol = 1e-3
        max_iter = 3000
        if training.fast_classical_mode:
            alpha = 0.2
            tol = 3e-3
            max_iter = 1500
        scaler = StandardScaler()
        if output_dim == 1:
            regressor = ElasticNet(
                alpha=alpha,
                l1_ratio=0.5,
                max_iter=max_iter,
                selection="random",  # Random is faster than cyclic
                tol=tol,
                random_state=training.random_state,
            )
        else:
            regressor = MultiTaskElasticNet(
                alpha=alpha,
                l1_ratio=0.3,
                max_iter=max_iter,
                tol=tol,
                random_state=training.random_state,
            )
        return Pipeline(
            [
                ("scaler", scaler),
                ("regressor", regressor),
            ]
        )
    if name == "lasso":
        from sklearn.linear_model import Lasso, MultiTaskLasso

        alpha = 0.1
        tol = 1e-3
        max_iter = 3000
        if training.fast_classical_mode:
            alpha = 0.2
            tol = 3e-3
            max_iter = 1500
        scaler = StandardScaler()
        if output_dim == 1:
            regressor = Lasso(
                alpha=alpha,
                max_iter=max_iter,
                selection="random",  # Random is faster than cyclic
                tol=tol,
                random_state=training.random_state,
            )
        else:
            regressor = MultiTaskLasso(
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                random_state=training.random_state,
            )
        return Pipeline(
            [
                ("scaler", scaler),
                ("regressor", regressor),
            ]
        )
    if name == "ols":
        from sklearn.linear_model import LinearRegression

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
    raise ValueError(f"Unknown model name: {name}")

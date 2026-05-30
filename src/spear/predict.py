import argparse
import json
from pathlib import Path
from typing import List

import anndata as ad
import joblib
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import TrainingConfig
from .data import (
    GeneInfo,
    PeakIndexer,
    _counts_per_million,
    _log1p_cpm,
    _tfidf_matrix,
    build_cellwise_features_only,
    parse_gtf,
    select_genes,
)

from .models import build_model, TorchModelBundle
from .training import _reshape_tensor_for_model


from .logging_utils import get_logger

_LOG = get_logger(__name__)


def _load_run_context(run_dir: Path) -> tuple[TrainingConfig, Path]:
    cfg_path = run_dir / "run_configuration.json"
    payload = json.loads(cfg_path.read_text())
    training_payload = payload["pipeline_config"]["training"]
    training = TrainingConfig(**training_payload)
    gtf_path_str = payload["pipeline_config"]["paths"]["gtf_path"]
    gtf_path = Path(gtf_path_str).expanduser()
    return training, gtf_path


def _load_selected_genes(run_dir: Path, gtf_path: Path) -> List[GeneInfo]:
    manifest = run_dir / "selected_genes.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"selected_genes.csv not found in {run_dir}")
    lines = manifest.read_text().splitlines()
    header = lines[0].split(",")
    col_idx = {name: idx for idx, name in enumerate(header)}
    gene_names = [row.split(",")[col_idx.get("gene_name", 0)] for row in lines[1:]]
    annotations = parse_gtf(gtf_path, gene_names=gene_names)
    return select_genes(annotations, requested_genes=gene_names, max_genes=None)


def _normalize_atac(atac: ad.AnnData, training: TrainingConfig) -> ad.AnnData:
    if training.atac_layer and training.atac_layer not in atac.layers:
        if training.atac_layer == "counts_per_million":
            atac.layers[training.atac_layer] = _counts_per_million(atac.X)
        elif training.atac_layer == "tfidf":
            atac.layers[training.atac_layer] = _tfidf_matrix(atac.X)
        elif training.atac_layer == "log1p_cpm":
            atac.layers[training.atac_layer] = _log1p_cpm(atac.X)
    return atac


def _knn_smooth_features(
    X: np.ndarray,
    cell_ids: np.ndarray,
    *,
    group_size: int,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    """Apply k-NN smoothing to inference features to mirror training."""

    if X.size == 0:
        return X

    n_cells = X.shape[0]
    if group_size <= 1 or n_cells <= 1:
        return X

    components = max(1, min(n_components, X.shape[1], n_cells))
    if components < 1:
        return X

    X_for_pca = X
    if X.shape[1] > 0:
        scaler = StandardScaler(with_mean=False)
        try:
            X_for_pca = scaler.fit_transform(X)
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning(
                "Inference smoothing: scaler failed (%s); using raw features", exc
            )

    try:
        pca = PCA(n_components=components, random_state=random_state)
        embedding = pca.fit_transform(X_for_pca)
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("Inference smoothing: PCA failed (%s); skipping smoothing", exc)
        return X

    k_neighbors = min(group_size - 1, n_cells - 1)
    if k_neighbors < 1:
        return X

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    nn.fit(embedding)
    _, neighbor_indices = nn.kneighbors(embedding)

    X_smoothed = np.zeros_like(X, dtype=np.float32)
    for i in range(n_cells):
        neighbor_set = neighbor_indices[
            i, : k_neighbors + 1
        ]  # includes the cell itself
        X_smoothed[i] = (
            np.asarray(X[neighbor_set], dtype=np.float64)
            .mean(axis=0)
            .astype(np.float32)
        )

    _LOG.info(
        "Inference smoothing applied | cells=%d | k=%d | components=%d",
        n_cells,
        k_neighbors,
        components,
    )
    return X_smoothed


def predict(
    run_dir: Path,
    model_name: str,
    atac_path: Path,
    output_path: Path | None = None,
) -> Path:
    training, gtf_path = _load_run_context(run_dir)
    genes = _load_selected_genes(run_dir, gtf_path)

    atac = ad.read_h5ad(atac_path.as_posix())
    atac = _normalize_atac(atac, training)
    peak_indexer = PeakIndexer(atac, layer=training.atac_layer)
    feature_data = build_cellwise_features_only(genes, atac, peak_indexer, training)

    model_dir = run_dir / "models" / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_meta_path = model_dir / "model_meta.json"
    model_meta: dict = {}
    if model_meta_path.exists():
        try:
            model_meta = json.loads(model_meta_path.read_text())
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("Failed to read model metadata %s: %s", model_meta_path, exc)
            model_meta = {}

    feature_block_indices = model_meta.get("feature_block_indices")
    if feature_block_indices is not None and not isinstance(
        feature_block_indices, list
    ):
        feature_block_indices = None

    feature_scaler_path = model_dir / "feature_scaler.pkl"
    target_scaler_path = model_dir / "target_scaler.pkl"
    feature_scaler = (
        joblib.load(feature_scaler_path) if feature_scaler_path.exists() else None
    )
    target_scaler = (
        joblib.load(target_scaler_path) if target_scaler_path.exists() else None
    )

    model_pt = model_dir / "model.pt"
    model_pkl = model_dir / "model.pkl"
    model = None
    reshape = None
    if model_pt.exists():
        state = torch.load(model_pt, map_location="cpu")
        reshape = state.get("reshape")
        bundle = build_model(
            model_name,
            feature_data.num_features(),
            training,
            output_dim=len(feature_data.genes),
            feature_block_indices=feature_block_indices,
        )
        if isinstance(bundle, TorchModelBundle):
            model = bundle.model
            model.load_state_dict(state["state_dict"])
            reshape = reshape or bundle.reshape
        else:
            raise RuntimeError(
                "Saved torch model but build_model did not return TorchModelBundle"
            )
    elif model_pkl.exists():
        model = joblib.load(model_pkl)
    else:
        raise FileNotFoundError("No saved model artifact found (.pt or .pkl)")

    X = feature_data.X
    if sp.issparse(X):
        X = X.toarray().astype(np.float32)
    if (
        training.enable_smoothing
        and training.smoothing_k > 1
        and getattr(training, "smoothing_target", "all_splits") == "all_splits"
    ):
        X = _knn_smooth_features(
            X,
            feature_data.cell_ids,
            group_size=training.smoothing_k,
            n_components=training.smoothing_pca_components,
            random_state=training.random_state,
        )
    if feature_scaler is not None:
        X = feature_scaler.transform(X)

    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            tens = torch.tensor(X, dtype=torch.float32)
            tens = _reshape_tensor_for_model(tens, reshape)
            preds = model(tens).cpu().numpy()
    else:
        preds = model.predict(X)

    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    if target_scaler is not None:
        preds = target_scaler.inverse_transform(preds)
    prediction_min = getattr(training, "prediction_min_value", 0.0)
    if prediction_min is not None:
        preds = np.maximum(preds, float(prediction_min))

    rows = []
    for cell_idx, cell_id in enumerate(feature_data.cell_ids):
        for gene_idx, gene in enumerate(feature_data.genes):
            rows.append(
                {
                    "cell_id": cell_id,
                    "gene": gene.gene_name,
                    "y_pred": float(preds[cell_idx, gene_idx]),
                }
            )

    df = pd.DataFrame(rows)
    out_path = output_path or (model_dir / "predictions_inference.csv")
    df.to_csv(out_path, index=False)
    _LOG.info(
        "Saved inference predictions for %d cells to %s",
        len(feature_data.cell_ids),
        out_path,
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on new ATAC AnnData using a trained model"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to training run directory (contains models/)",
    )
    parser.add_argument(
        "--model", required=True, help="Model name under run_dir/models/"
    )
    parser.add_argument(
        "--atac-path", required=True, help="Path to ATAC AnnData (.h5ad) for inference"
    )
    parser.add_argument("--output", help="Optional output CSV path for predictions")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    atac_path = Path(args.atac_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    predict(run_dir, args.model, atac_path, output_path)


if __name__ == "__main__":
    main()

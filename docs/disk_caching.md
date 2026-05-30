# Disk Caching for SPEAR Preprocessing

## Overview

Accelerate repeated model training by persisting expensive preprocessing computations to disk. Train multiple models on the same dataset without recomputing splits, scaling, or transformations.

When training multiple models with the same preprocessing configuration, expensive operations (data loading, train/val/test splitting, feature scaling, smoothing, pseudobulk aggregation) are unnecessarily recomputed. The caching system:

- Computes preprocessing once per unique configuration
- Persists splits and scalers to disk in `data/.spear_cache/`
- Reuses cached data for subsequent model training (30–100x preprocessing speedup)
- Supports both per-gene and cell-wise preprocessing caches; the default training mode is per-gene, while multi-output remains available when explicitly enabled.

## Inputs

- Preprocessed AnnData files for RNA/ATAC.
- `TrainingConfig` values that define the preprocessing pipeline.
- Cache directory, typically `data/.spear_cache/`.

## Outputs

- Cached splits (`.npz`) and scalers (`.pkl`, JSON fallback) under `data/.spear_cache/` (sparse feature matrices are stored in separate `.npz` files).
- Deterministic cache keys that map a preprocessing configuration to disk artifacts.

## Usage

### Orientation

### Computational frame

- Preprocessing pipelines (loading, splitting, scaling, smoothing) are memory and I/O intensive and identical across runs with the same configuration.
- Hash-based cache keys ensure deterministic reuse: same config → same cache file, different config → separate cache (no conflicts).
- Three-tier lookup (in-memory → disk → compute) balances speed with memory efficiency across sessions and experiments.

### Key data products

- Embryonic dataset: 1000 genes from manifest (`data/embryonic/manifests/1000_random_genes.csv`), 54,301 cells after QC (see `docs/mouse_esc_dataset.md`).
- Endothelial dataset: 1000 randomly selected genes, 4,735 cells after QC (see `docs/endothelial_dataset.md`).
- Cached splits and scalers: `data/.spear_cache/<config_hash>_cellwise_splits.npz` and `*_cellwise_scalers.pkl` (JSON fallback supported).
- CLI cache scope: `--cache-dir` defaults to `--cache-scope auto`, which caches cell-wise/multi-output preprocessing only. Per-gene disk caching writes one split file per gene and must be enabled explicitly with `--cache-scope gene` or `--cache-scope all`.

### How Caching Works

#### Cache Key Generation

The cache key is a SHA1 hash of the preprocessing configuration:

```python
cache_key = SHA1(json.dumps({
    "scope": "cellwise",
    "train_fraction": 0.6,
    "val_fraction": 0.2,
    "test_fraction": 0.2,
    "random_state": 42,
    "scaler": "standard",
    "target_scaler": "standard",
    "enable_smoothing": False,
    "smoothing_k": 1,
    "global_atac_components": 0,
    "pseudobulk_group_size": 1,
    # ...other config params
}, sort_keys=True))
# Result: "a1b2c3d4e5..." (40 character hex)
```

**Key principle**: Same configuration → Same hash → Same cache file.

#### Three-Tier Lookup

When `prepare_cellwise_data(dataset, config, cache_dir)` is called:

```text
1. Check in-memory cache
   └─ Return if found (5 ms)

2. Check disk cache at: data/.spear_cache/<hash>_cellwise_splits.npz
   └─ Load and return if found (5–10 sec)

3. Compute preprocessing
   └─ Compute, save to disk, return (5–10 min)
```

#### Cache Files and Formats

```text
data/.spear_cache/
├── a1b2c3d4e5_cellwise_splits.npz      # Train/val/test splits (NumPy compressed)
├── a1b2c3d4e5_cellwise_scalers.pkl     # Fitted StandardScaler/MinMaxScaler (Pickle)
├── a1b2c3d4e5_cellwise_X_train.npz     # Sparse feature matrix (if applicable)
├── a1b2c3d4e5_cellwise_X_val.npz
├── a1b2c3d4e5_cellwise_X_test.npz
├── f6g7h8i9j0_cellwise_splits.npz      # Different config, separate cache
└── f6g7h8i9j0_cellwise_scalers.pkl
```

**Format rationale:**

- `.npz` — NumPy compressed arrays; fast loading (5–10 sec for large datasets), standardized binary format, cross-platform compatible.
- `.pkl` — Pickle-serialized scikit-learn scalers; preserves fitted parameters (mean, scale, etc.) and internal state.
- `.json` — JSON scalers still load if present (fallback).

### Setup and Practical Steps

#### Step 1 – Build Processed Inputs

```bash
cd /path/to/SPEAR
python scripts/download_geo_raw_data.py --datasets embryonic endothelial
python scripts/preprocess_geo_raw_to_spear.py --datasets embryonic
python scripts/preprocess_geo_raw_to_spear.py --datasets endothelial
python scripts/generate_all_sample_manifests.py --base-dir data --gene-count 1000 --random-state 42
```

These scripts:

- download raw GEO inputs and shared references into `data/`
- preprocess each sample into paired RNA/ATAC `.h5ad` files under `data/*/processed/per_sample/`
- generate per-sample gene manifests under `data/*/manifests/`

The actual cached cell-wise preprocessing happens automatically at training time when `--cache-dir` is enabled. In per-gene mode, the same `--cache-dir` does not write per-gene disk caches unless `--cache-scope gene` or `--cache-scope all` is provided.

**Expected output:** processed per-sample `.h5ad` files, manifests, and then cell-wise cache files appearing under `data/.spear_cache/` after the first multi-output training run with `--cache-dir`.

#### Step 2 – Train Models

Train models using cached preprocessing:

```bash
# First model: cache lookup, then compute (if needed)
python -m spear.cli train --dataset embryonic --models ridge --cache-dir data/.spear_cache

# Subsequent models: preprocessing loads from cache in 5–10 sec
python -m spear.cli train --dataset embryonic --models elasticnet --cache-dir data/.spear_cache
python -m spear.cli train --dataset endothelial --models ridge --cache-dir data/.spear_cache
```

To deliberately persist per-gene preprocessing across separate jobs or reruns:

```bash
python -m spear.cli --models ridge --gene-manifest data/embryonic/manifests/1000_random_genes.csv --cache-dir data/.spear_cache --cache-scope gene
```

Use this sparingly: per-gene disk caching can produce one large `*_gene_splits.npz` per gene and preprocessing configuration.

### API Usage

#### Python Integration

Use caching in Python workflows:

```python
from pathlib import Path
from spear.training import prepare_cellwise_data
from spear.config import TrainingConfig

cache_dir = Path("data/.spear_cache")
config = TrainingConfig()

# First call: Computes & caches (5–10 min)
prepared = prepare_cellwise_data(dataset, config, cache_dir=cache_dir)

# Second call: Loads from cache (5–10 sec)
prepared = prepare_cellwise_data(dataset, config, cache_dir=cache_dir)
```

#### Configuration

Modify preprocessing parameters in the CLI config or the underlying training configuration:

```python
config = TrainingConfig(
    train_fraction=0.6,           # 60% training data
    val_fraction=0.2,             # 20% validation data
    test_fraction=0.2,            # 20% test data
    random_state=42,              # Reproducibility seed
    scaler="standard",            # StandardScaler or MinMaxScaler
    target_scaler="standard",     # Scaler for multi-target y
    enable_smoothing=False,       # KNN smoothing (optional)
    smoothing_k=5,                # Neighborhood size if enabled
    pseudobulk_group_size=1,      # 1 = no aggregation, >1 = aggregate cells
)
```

**Principle**: Different config → Different hash → Different cache file (no conflicts).

#### What Gets Cached

```python
{
    "splits": {
        "X_train": np.ndarray,      # Features (cells × features)
        "X_val": np.ndarray,
        "X_test": np.ndarray,
        "y_train": np.ndarray,      # Targets (cells × genes) - multi-output!
        "y_val": np.ndarray,
        "y_test": np.ndarray,
        "cell_ids_train/val/test": np.ndarray,      # Cell identifiers
        "group_labels_train/val/test": np.ndarray,  # Batch/sample groups
        "X_train_raw": np.ndarray,  # Unscaled (for cross-validation)
        "y_train_raw": np.ndarray,  # Unscaled (for cross-validation)
    },
    "feature_scaler": StandardScaler,  # Fitted on X_train
    "target_scaler": StandardScaler,   # Fitted on y_train
}
```

**Not cached** (recomputed each run): model weights, predictions, feature importances, training history.

### Performance

#### Speed Comparison

| Operation                   | Time     | Speedup     |
| --------------------------- | -------- | ----------- |
| First preprocessing run     | 5–10 min | Baseline    |
| Cached preprocessing load   | 5–10 sec | 30–100x     |
| Model training (ridge)      | 1-2 min  | (unchanged) |
| Model training (neural net) | 5-15 min | (unchanged) |

#### Multiple Models on Same Dataset

```text
Without cache:
  10 models × (10 min prep + 10 min train) = 200 minutes

With cache:
  10 min prep (first model) + 10 × 10 min train = 110 minutes
  Speedup: 1.8x overall, 30-100x on preprocessing
```

### Advanced Topics and Troubleshooting

#### Regenerating Cache

To clear and rebuild cache:

```bash
# Remove old cache
rm -rf data/.spear_cache

# Regenerate with fresh preprocessing on the next training run
python -m spear.cli --cache-dir data/.spear_cache ...
```

#### Multiple Configurations

Different preprocessing configs are cached independently:

```python
# Config 1: Default (60/20/20 split)
config1 = TrainingConfig(train_fraction=0.6)
prepared1 = prepare_cellwise_data(dataset, config1, cache_dir)

# Config 2: Different split (70/15/15)
config2 = TrainingConfig(train_fraction=0.7)
prepared2 = prepare_cellwise_data(dataset, config2, cache_dir)

# Each gets its own cache file due to different hash
```

#### Memory Efficiency

For extremely large datasets:

1. **Increase pseudobulk aggregation**: `pseudobulk_group_size=10` (reduces sample count before scaling).
2. **Enable smoothing**: `enable_smoothing=True, smoothing_k=5` (further reduces effective sample size).
3. **Reduce features**: Edit manifest to include fewer genes.

#### Cache Miss Detection

**Symptom**: Preprocessing still takes 5–10 minutes despite cache.

**Solutions:**

1. Verify `--cache-dir data/.spear_cache` parameter is passed to training script.
2. Check that preprocessing config matches (same train/val/test split, scaler, etc.).
3. Verify cache files exist: `ls -la data/.spear_cache/`.
4. Check logs for: "Loaded prepared cell-wise data from data/.spear_cache".

#### Stale Cache (Outdated Preprocessing)

**Solution:**

```bash
rm -rf data/.spear_cache
python -m spear.cli --cache-dir data/.spear_cache ...
```

#### Out of Memory During Preprocessing

If densifying sparse matrices fails:

1. Increase pseudobulk aggregation: `pseudobulk_group_size=10`.
2. Enable KNN smoothing: `enable_smoothing=True, smoothing_k=5`.
3. Reduce feature count in manifest.

#### Pickle Compatibility Error

**Symptom**: "Cannot unpickle" error when loading scalers.

**Cause**: Python or scikit-learn version mismatch.

**Solution**:

```bash
# Remove pickle files (NPZ arrays will still load)
rm data/.spear_cache/*.pkl

# Rerun training to recompute scalers
python -m spear.cli --cache-dir data/.spear_cache ...
```

## References

- Implementation details: `src/spear/cache.py`, `src/spear/training.py`
- Dataset details: `docs/mouse_esc_dataset.md`, `docs/endothelial_dataset.md`
- Configuration reference: `docs/config_reference.md`

## Implementation Details

### File Structure

```text
spear/
├── src/spear/
│   ├── cache.py                    # Serialization/deserialization functions
│   └── training.py                 # Modified for caching support
└── docs/
    └── disk_caching.md             # This documentation
```

### Cache Output

```text
data/.spear_cache/
├── a1b2c3d4e5_cellwise_splits.npz      # Train/val/test splits (NumPy compressed)
├── a1b2c3d4e5_cellwise_scalers.pkl     # Fitted StandardScaler/MinMaxScaler (Pickle)
├── f6g7h8i9j0_cellwise_splits.npz      # Different config, separate cache
└── f6g7h8i9j0_cellwise_scalers.pkl
```

### Serialization Strategy

- **NumPy (.npz)**: Standard binary format, efficient compression, fast I/O, cross-platform.
- **Pickle (.pkl)**: Preserves scikit-learn scaler state; version-dependent (regeneration may be needed after sklearn updates).

### Cache Key Components

Hash includes these configuration parameters:

```python
{
    "scope": "cellwise" | "gene",
    "train_fraction": float,
    "val_fraction": float,
    "test_fraction": float,
    "random_state": int,
    "group_key": str | None,
    "enable_smoothing": bool,
    "smoothing_k": int,
    "smoothing_pca_components": int,
    "global_atac_components": int,
    "pseudobulk_group_size": int,
    "pseudobulk_pca_components": int,
    "scaler": "standard" | "minmax" | None,
    "target_scaler": "standard" | "minmax" | None,
}
```

**Advantage**: Same configuration always maps to same cache file (deterministic, no conflicts).

## Reference

For more information on preprocessing configuration, see [config_reference.md](config_reference.md). For dataset details, see [mouse_esc_dataset.md](mouse_esc_dataset.md) (embryonic) or [endothelial_dataset.md](endothelial_dataset.md) (endothelial).

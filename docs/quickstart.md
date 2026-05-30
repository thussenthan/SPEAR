# Quickstart

This guide covers the current sample-first workflow under `data/`:

1. install dependencies
2. run a tiny synthetic smoke test
3. download fresh raw data and references from the original sources
4. preprocess samples into SPEAR-ready RNA/ATAC `.h5ad` files
5. generate three 1000-gene manifest types per sample
6. run a single SPEAR experiment

## What the endothelial pipeline currently does

The `endothelial` dataset label is just the folder name for `GSE270141`.

The current pipeline does **not** subset to a curated endothelial cell-type population. It processes all cells in the two published Multiome samples:

- `S3H_hypoxia`
- `S3N_normoxia`

Those samples come from the GEO study and are split by oxygen condition, not by a downstream cell-type filter. If you want a true endothelial-only subset, that would need an additional annotation and filtering step.

## 1. Install dependencies

Run from the repo root:

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Tiny smoke test without downloading real data

Create a tiny synthetic paired ATAC/RNA dataset, GTF, and gene manifest:

```bash
python scripts/create_tiny_example_data.py
```

Run a minimal per-gene ridge model on the generated files:

```bash
spear \
  --rna-path data/examples/tiny/tiny_RNA.h5ad \
  --atac-path data/examples/tiny/tiny_ATAC.h5ad \
  --gtf-path data/examples/tiny/tiny.gtf \
  --gene-manifest data/examples/tiny/tiny_genes.csv \
  --models ridge \
  --device cpu \
  --k-folds 1 \
  --disable-smoothing \
  --min-cells-per-gene 5 \
  --run-name tiny_smoke_ridge
```

Expected outputs include `output/results/tiny_smoke_ridge/summary_metrics.csv`,
`summary_metrics_per_gene.csv`, `run_configuration.json`, and model-level files under
`output/results/tiny_smoke_ridge/models/ridge/`.

The tiny files are generated under ignored `data/` paths and are intended only to verify installation and CLI behavior. They are not biological data.

## 3. Download raw data and references from scratch

Download everything for both datasets, plus both GTF references and both ATAC peak references:

```bash
python scripts/download_geo_raw_data.py --datasets embryonic endothelial
```

Download only one embryonic replicate from scratch, while still downloading the shared references:

```bash
python scripts/download_geo_raw_data.py --datasets embryonic --samples E7.5_rep1
```

Download only the endothelial dataset from scratch:

```bash
python scripts/download_geo_raw_data.py --datasets endothelial
```

Notes:

- the default is fresh download from source
- local cache reuse only happens if you explicitly pass `--use-local-cache`
- downloaded references land in `data/references/`

## 4. Preprocess into per-sample SPEAR-ready files

Preprocess all embryonic and endothelial samples separately:

```bash
python scripts/preprocess_geo_raw_to_spear.py --datasets embryonic
python scripts/preprocess_geo_raw_to_spear.py --datasets endothelial
```

Preprocess just one embryonic replicate:

```bash
python scripts/preprocess_geo_raw_to_spear.py --datasets embryonic --samples E7.5_rep1
```

Preprocess just one endothelial sample:

```bash
python scripts/preprocess_geo_raw_to_spear.py --datasets endothelial --samples S3H_hypoxia
```

Outputs go to:

- `data/embryonic/processed/per_sample/<sample>/`
- `data/endothelial/processed/per_sample/<sample>/`

Each processed sample should contain:

- `<sample>_RNA_qc.h5ad`
- `<sample>_ATAC_qc.h5ad`

## 5. Generate manifests

Generate all three 1000-gene manifest types for every processed sample:

```bash
python scripts/generate_all_sample_manifests.py --base-dir data --gene-count 1000 --random-state 42
```

Generate manifests for just one embryonic replicate:

```bash
python scripts/generate_all_sample_manifests.py \
  --base-dir data \
  --datasets embryonic \
  --samples E7.5_rep1 \
  --gene-count 1000 \
  --random-state 42
```

This creates, for each sample:

- `<sample>_random_1000.csv`
- `<sample>_hvg_1000.csv`
- `<sample>_low_noisy_1000.csv`

Outputs go to:

- `data/embryonic/manifests/`
- `data/endothelial/manifests/`

## 6. Run one SPEAR job

A minimal CPU example for one embryonic replicate using the random 1000-gene manifest:

```bash
spear \
  --rna-path data/embryonic/processed/per_sample/E7.5_rep1/E7.5_rep1_RNA_qc.h5ad \
  --atac-path data/embryonic/processed/per_sample/E7.5_rep1/E7.5_rep1_ATAC_qc.h5ad \
  --gtf-path data/references/GCF_000001635.27_genomic.gtf \
  --gene-manifest data/embryonic/manifests/E7.5_rep1_random_1000.csv \
  --models ridge \
  --device cpu \
  --k-folds 0 \
  --run-name e75_rep1_random1000_ridge
```

The same per-gene workflow can use peak-level cis features instead of aggregated bins:

```bash
spear \
  --rna-path data/embryonic/processed/per_sample/E7.5_rep1/E7.5_rep1_RNA_qc.h5ad \
  --atac-path data/embryonic/processed/per_sample/E7.5_rep1/E7.5_rep1_ATAC_qc.h5ad \
  --gtf-path data/references/GCF_000001635.27_genomic.gtf \
  --gene-manifest data/embryonic/manifests/E7.5_rep1_random_1000.csv \
  --per-gene \
  --per-gene-feature-basis peak \
  --per-gene-peak-min-peaks 10 \
  --window-bp 100000 \
  --models ridge \
  --device cpu \
  --run-name e75_rep1_random1000_ridge_peak100kb
```

For peak-only runs, generate or reuse manifests that were filtered with the same peak window and a nonzero `min_peaks_per_gene`. This prevents zero-peak genes from silently changing the feature basis in downstream sweeps.

A minimal CPU example for one endothelial sample using the HVG manifest:

```bash
spear \
  --rna-path data/endothelial/processed/per_sample/S3H_hypoxia/S3H_hypoxia_RNA_qc.h5ad \
  --atac-path data/endothelial/processed/per_sample/S3H_hypoxia/S3H_hypoxia_ATAC_qc.h5ad \
  --gtf-path data/references/gencode.v44.annotation.gtf.gz \
  --gene-manifest data/endothelial/manifests/S3H_hypoxia_hvg_1000.csv \
  --models ridge \
  --device cpu \
  --run-name s3h_hypoxia_hvg1000_ridge
```

## 7. Optional one-shot commands

Download, preprocess, and make manifests for one embryonic replicate:

```bash
python scripts/download_geo_raw_data.py --datasets embryonic --samples E7.5_rep1
python scripts/preprocess_geo_raw_to_spear.py --datasets embryonic --samples E7.5_rep1
python scripts/generate_all_sample_manifests.py \
  --base-dir data \
  --datasets embryonic \
  --samples E7.5_rep1 \
  --gene-count 1000 \
  --random-state 42 \
  --peak-window-bp 250000 \
  --min-peaks-per-gene 10
```

Download, preprocess, and make manifests for all samples:

```bash
python scripts/download_geo_raw_data.py --datasets embryonic endothelial
python scripts/preprocess_geo_raw_to_spear.py --datasets embryonic
python scripts/preprocess_geo_raw_to_spear.py --datasets endothelial
python scripts/generate_all_sample_manifests.py --base-dir data --gene-count 1000 --random-state 42
```

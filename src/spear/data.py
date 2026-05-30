from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import sparsefuncs

from .config import PathsConfig, TrainingConfig
from .logging_utils import get_logger

_LOG = get_logger(__name__)

_REFSEQ_TO_CHR = {
    "NC_000067": "chr1",
    "NC_000068": "chr2",
    "NC_000069": "chr3",
    "NC_000070": "chr4",
    "NC_000071": "chr5",
    "NC_000072": "chr6",
    "NC_000073": "chr7",
    "NC_000074": "chr8",
    "NC_000075": "chr9",
    "NC_000076": "chr10",
    "NC_000077": "chr11",
    "NC_000078": "chr12",
    "NC_000079": "chr13",
    "NC_000080": "chr14",
    "NC_000081": "chr15",
    "NC_000082": "chr16",
    "NC_000083": "chr17",
    "NC_000084": "chr18",
    "NC_000085": "chr19",
    "NC_000086": "chrX",
    "NC_000087": "chrY",
    "NC_005089": "chrM",
}


@dataclass
class GeneInfo:
    gene_id: str
    gene_name: str
    chrom: str
    tss: int
    strand: str


@dataclass
class GeneDataset:
    gene: GeneInfo
    X: np.ndarray
    y: np.ndarray
    cell_ids: np.ndarray
    feature_names: List[str]
    group_labels: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)
    prepared_cache: Dict[str, object] = field(default_factory=dict, repr=False)

    def num_cells(self) -> int:
        return int(self.X.shape[0])

    def num_features(self) -> int:
        return int(self.X.shape[1])


def _bin_label(bin_idx: int, training: TrainingConfig, gene: GeneInfo) -> str:
    """Label bins relative to the gene TSS, oriented by strand."""
    start_offset = (bin_idx * training.bin_size_bp) - training.window_bp
    end_offset = ((bin_idx + 1) * training.bin_size_bp) - training.window_bp
    if gene.strand == "-":
        start_offset, end_offset = -end_offset, -start_offset
    return f"bin_{int(start_offset)}_to_{int(end_offset)}"


def _orient_peak_window(
    indices: np.ndarray,
    midpoints: np.ndarray,
    gene: GeneInfo,
) -> tuple[np.ndarray, np.ndarray]:
    """Orient peak windows relative to transcription direction."""
    if gene.strand == "-":
        return indices[::-1], midpoints[::-1]
    return indices, midpoints


def _signed_peak_offset(midpoint: int, gene: GeneInfo) -> int:
    """Return a transcription-oriented peak midpoint offset from the TSS."""
    genomic_offset = int(midpoint) - int(gene.tss)
    if gene.strand == "-":
        return -genomic_offset
    return genomic_offset


def _compute_gene_features_all_cells(
    gene: GeneInfo,
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> Tuple[np.ndarray, List[str]]:
    start = gene.tss - training.window_bp
    end = gene.tss + training.window_bp
    indices, midpoints = peak_indexer.get_peaks_in_window(gene.chrom, start, end)
    num_bins = int(np.ceil((end - start) / training.bin_size_bp))
    if num_bins <= 0:
        raise ValueError("Invalid bin configuration")

    features = np.zeros((peak_indexer.n_cells, num_bins), dtype=np.float32)
    if indices.size == 0:
        return features, [
            f"bin_{(i * training.bin_size_bp) - training.window_bp}_to_{((i + 1) * training.bin_size_bp) - training.window_bp}"
            for i in range(num_bins)
        ]

    matrix = peak_indexer.matrix[:, indices]
    if sp.issparse(matrix):
        matrix = matrix.tocsc()

    local_mid = midpoints - start
    bin_ids = np.clip((local_mid // training.bin_size_bp).astype(int), 0, num_bins - 1)
    unique_bins = np.unique(bin_ids)
    for b in unique_bins:
        col_mask = bin_ids == b
        cols = np.where(col_mask)[0]
        if cols.size == 0:
            continue
        if sp.issparse(matrix):
            sub = matrix[:, cols]
            summed = np.asarray(sub.sum(axis=1)).ravel()
        else:
            sub = matrix[:, cols]
            summed = np.sum(sub, axis=1)
        features[:, b] = summed

    # Orient bins relative to transcription direction: negative = upstream, positive = downstream
    if gene.strand == "-":
        features = features[:, ::-1]
        order = list(reversed(range(num_bins)))
    else:
        order = list(range(num_bins))
    feature_names = [_bin_label(i, training, gene) for i in order]

    return features, feature_names


def _compute_gene_peak_features_all_cells(
    gene: GeneInfo,
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> Tuple[np.ndarray, List[str]]:
    """Return one-feature-per-peak ATAC accessibility in the TSS window for all cells.

    Unlike `_compute_gene_features_all_cells`, this preserves peak-level resolution
    instead of aggregating into fixed-width bins. Feature order is oriented relative to
    transcription direction (upstream features first on both strands).

    Returns a zero-column matrix when no peaks exist in the window; the caller should
    fall back to bin-based features in that case.
    """
    start = gene.tss - training.window_bp
    end = gene.tss + training.window_bp
    indices, midpoints = peak_indexer.get_peaks_in_window(gene.chrom, start, end)

    if indices.size == 0:
        return np.zeros((peak_indexer.n_cells, 0), dtype=np.float32), []

    indices_ord, midpoints_ord = _orient_peak_window(indices, midpoints, gene)
    matrix = peak_indexer.matrix[:, indices_ord]
    if sp.issparse(matrix):
        features = np.asarray(matrix.toarray(), dtype=np.float32)
    else:
        features = np.asarray(matrix, dtype=np.float32)

    feature_names: List[str] = []
    for idx, mid in zip(indices_ord.tolist(), midpoints_ord.tolist()):
        peak_id = (
            peak_indexer.peak_ids[int(idx)]
            if int(idx) < peak_indexer.n_peaks
            else f"peak_{int(mid)}"
        )
        offset = _signed_peak_offset(int(mid), gene)
        feature_names.append(f"{peak_id}|offset_{offset:+d}bp")

    return features, feature_names


@dataclass
class CellwiseDataset:
    genes: List[GeneInfo]
    X: np.ndarray | sp.spmatrix
    y: np.ndarray
    cell_ids: np.ndarray
    feature_names: List[str]
    group_labels: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)
    prepared_cache: Dict[str, object] = field(default_factory=dict, repr=False)
    feature_block_slices: List[Tuple[int, int]] = field(default_factory=list)
    feature_block_indices: List[np.ndarray] = field(default_factory=list)

    def num_cells(self) -> int:
        return int(self.X.shape[0])

    def num_features(self) -> int:
        return int(self.X.shape[1])

    def num_genes(self) -> int:
        return len(self.genes)


def preprocess_modalities(
    atac: ad.AnnData,
    rna: ad.AnnData,
    training: TrainingConfig,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Apply modality-specific normalization layers."""

    if training.atac_layer and training.atac_layer not in atac.layers:
        if training.atac_layer == "tfidf":
            atac.layers[training.atac_layer] = _tfidf_matrix(atac.X)
        elif training.atac_layer == "counts_per_million":
            atac.layers[training.atac_layer] = _counts_per_million(atac.X)
        elif training.atac_layer == "log1p_cpm":
            atac.layers[training.atac_layer] = _log1p_cpm(atac.X)
        else:
            _LOG.warning(
                "Unknown ATAC layer requested: %s; skipping normalization",
                training.atac_layer,
            )

    if (
        training.rna_expression_layer
        and training.rna_expression_layer not in rna.layers
    ):
        if training.rna_expression_layer == "log1p_cpm":
            rna.layers[training.rna_expression_layer] = _log1p_cpm(rna.X)
        else:
            _LOG.warning(
                "Unknown RNA layer requested: %s; skipping normalization",
                training.rna_expression_layer,
            )

    return atac, rna


def _counts_per_million(matrix: sp.spmatrix | np.ndarray) -> sp.spmatrix | np.ndarray:
    """Scale counts to counts-per-million per cell."""

    if sp.issparse(matrix):
        counts = matrix.tocsr().astype(np.float32, copy=True)
        totals = np.asarray(counts.sum(axis=1)).ravel()
        totals[totals == 0] = 1.0
        inv_totals = (1e6 / totals).astype(np.float32)
        counts = counts.tocoo()
        counts.data *= inv_totals[counts.row]
        return counts.tocsr()

    counts_arr = np.asarray(matrix, dtype=np.float32)
    totals = counts_arr.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return counts_arr * (1e6 / totals)


def _log1p_cpm(matrix: sp.spmatrix | np.ndarray) -> sp.spmatrix | np.ndarray:
    """Apply counts-per-million normalization followed by log1p transform."""

    cpm = _counts_per_million(matrix)
    if sp.issparse(cpm):
        data = np.log1p(cpm.data.astype(np.float64, copy=False))
        cpm.data = data.astype(np.float32, copy=False)
        return cpm.tocsr()

    dense = np.asarray(cpm, dtype=np.float64)
    dense = np.log1p(dense)
    return dense.astype(np.float32)


def _tfidf_matrix(matrix: sp.spmatrix | np.ndarray) -> sp.spmatrix | np.ndarray:
    """TF-IDF with per-row L2 normalization (matches ArchR/Signac LSI convention).

    Steps: TF (row-normalize counts) → IDF weighting → L2-normalize each cell row.
    L2 normalization ensures all cells land on the unit hypersphere, removing the
    residual magnitude differences that remain after TF alone.
    """

    if sp.issparse(matrix):
        counts = matrix.tocsr().astype(np.float32, copy=True)
        n_cells = counts.shape[0]

        row_totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        row_totals[row_totals == 0] = 1.0
        sparsefuncs.inplace_row_scale(counts, 1.0 / row_totals)

        tfidf = counts.tocsc()
        doc_freq = np.diff(tfidf.indptr).astype(np.float32)
        idf = np.log1p(n_cells / (1.0 + doc_freq))
        sparsefuncs.inplace_column_scale(tfidf, idf.astype(np.float32))
        tfidf = tfidf.tocsr()

        # L2-normalize each row so all cells have unit norm
        row_norms = np.sqrt(np.asarray(tfidf.power(2).sum(axis=1)).ravel()).astype(
            np.float32
        )
        row_norms[row_norms == 0] = 1.0
        sparsefuncs.inplace_row_scale(tfidf, 1.0 / row_norms)
        return tfidf

    counts_arr = np.asarray(matrix, dtype=np.float32)
    totals = counts_arr.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    tf = counts_arr / totals
    df = (counts_arr > 0).sum(axis=0, keepdims=False).astype(np.float32)
    idf = np.log1p(counts_arr.shape[0] / (1.0 + df))
    tfidf = (tf * idf).astype(np.float32)
    row_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    return (tfidf / row_norms).astype(np.float32)


class PeakIndexer:
    """Helper that indexes ATAC peaks by chromosome for fast lookup."""

    def __init__(self, atac: ad.AnnData, layer: Optional[str] = None):
        var_df = atac.var.copy()

        chroms, starts, ends = _extract_peak_coordinates(var_df)

        midpoints = (starts + ends) // 2

        self.chrom_to_midpoints: Dict[str, np.ndarray] = {}
        self.chrom_to_indices: Dict[str, np.ndarray] = {}
        unique_chroms = pd.unique(chroms)
        for chrom in unique_chroms:
            mask = chroms == chrom
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            mids = midpoints[idx]
            order = np.argsort(mids, kind="mergesort")
            self.chrom_to_midpoints[chrom] = mids[order]
            self.chrom_to_indices[chrom] = idx[order]

        if layer and layer in atac.layers:
            matrix = atac.layers[layer]
            self.layer = layer
        else:
            matrix = atac.X
            self.layer = None
        if sp.issparse(matrix):
            self.matrix = matrix.tocsr()
        else:
            self.matrix = np.asarray(matrix, dtype=np.float32)
        self.n_cells = atac.n_obs
        self.peak_ids = np.asarray(var_df.index).astype(str)
        self.n_peaks = int(self.peak_ids.shape[0])
        self._global_feature_cache: Dict[
            Tuple[int, int, str | None, Tuple[int, int]], Tuple[np.ndarray, List[str]]
        ] = {}

    def get_peaks_in_window(
        self, chrom: str, start: int, end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        midpoints = self.chrom_to_midpoints.get(chrom)
        indices = self.chrom_to_indices.get(chrom)
        if midpoints is None or indices is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        left = np.searchsorted(midpoints, start, side="left")
        right = np.searchsorted(midpoints, end, side="right")
        local = indices[left:right]
        local_midpoints = midpoints[left:right]
        return local, local_midpoints


def _compute_global_atac_features(
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> Tuple[np.ndarray, List[str]]:
    """Return optional global ATAC cell-state components for all cells."""

    requested = int(getattr(training, "global_atac_components", 0) or 0)
    if requested <= 0:
        return np.zeros((peak_indexer.n_cells, 0), dtype=np.float32), []

    cache_key = (
        requested,
        int(getattr(training, "random_state", 42)),
        peak_indexer.layer,
        tuple(int(v) for v in peak_indexer.matrix.shape),
    )
    cached = peak_indexer._global_feature_cache.get(cache_key)
    if cached is not None:
        return cached

    max_components = min(
        requested, max(0, peak_indexer.n_cells - 1), peak_indexer.n_peaks
    )
    names = [f"global_atac_svd_{i + 1}" for i in range(requested)]
    if max_components <= 0:
        features = np.zeros((peak_indexer.n_cells, requested), dtype=np.float32)
        result = (features, names)
        peak_indexer._global_feature_cache[cache_key] = result
        return result

    svd = TruncatedSVD(n_components=max_components, random_state=training.random_state)
    features = np.asarray(svd.fit_transform(peak_indexer.matrix), dtype=np.float32)
    if max_components < requested:
        padding = np.zeros(
            (peak_indexer.n_cells, requested - max_components), dtype=np.float32
        )
        features = np.hstack([features, padding])

    result = (features, names)
    peak_indexer._global_feature_cache[cache_key] = result
    return result


def _append_global_features(
    matrix: np.ndarray | sp.spmatrix,
    feature_names: List[str],
    feature_block_indices: Optional[List[np.ndarray]],
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> Tuple[np.ndarray | sp.spmatrix, List[str], Optional[List[np.ndarray]]]:
    global_features, global_names = _compute_global_atac_features(
        peak_indexer, training
    )
    if global_features.shape[1] == 0:
        return matrix, feature_names, feature_block_indices

    offset = int(matrix.shape[1])
    if sp.issparse(matrix):
        matrix_out: np.ndarray | sp.spmatrix = sp.hstack(
            [matrix, sp.csr_matrix(global_features)],
            format="csr",
        )
    else:
        matrix_out = np.hstack(
            [np.asarray(matrix, dtype=np.float32), global_features]
        ).astype(np.float32)

    names_out = list(feature_names) + list(global_names)
    if feature_block_indices is None:
        return matrix_out, names_out, None

    global_indices = np.arange(
        offset, offset + global_features.shape[1], dtype=np.int64
    )
    blocks_out: List[np.ndarray] = []
    for block in feature_block_indices:
        block_arr = np.asarray(block, dtype=np.int64).ravel()
        block_arr = block_arr[block_arr >= 0]
        blocks_out.append(np.concatenate([block_arr, global_indices]))
    return matrix_out, names_out, blocks_out


def load_datasets(paths: PathsConfig) -> Tuple[ad.AnnData, ad.AnnData]:
    _LOG.info("Loading AnnData objects from %s and %s", paths.atac_path, paths.rna_path)
    atac = ad.read_h5ad(paths.atac_path.as_posix())
    rna = ad.read_h5ad(paths.rna_path.as_posix())

    if atac.n_obs != rna.n_obs:
        raise ValueError(
            "ATAC and RNA AnnData objects have different numbers of cells; harmonized pairs expected"
        )

    atac_cells = np.asarray(atac.obs_names).astype(str)
    rna_cells = np.asarray(rna.obs_names).astype(str)
    if not np.array_equal(atac_cells, rna_cells):
        _LOG.info("Aligning ATAC/RNA data to shared barcodes")
        shared = np.intersect1d(atac_cells, rna_cells)
        if shared.size == 0:
            raise ValueError("No overlapping cell barcodes between ATAC and RNA data")
        atac = atac[shared].copy()
        rna = rna[shared].copy()
    return atac, rna


def parse_gtf(
    gtf_path: Path,
    chromosomes: Optional[Sequence[str]] = None,
    gene_names: Optional[Sequence[str]] = None,
) -> List[GeneInfo]:
    _LOG.info("Parsing GTF annotations from %s", gtf_path)
    opener = gzip.open if gtf_path.suffix == ".gz" else open
    records: List[GeneInfo] = []
    target_chroms_raw = set(chromosomes) if chromosomes else None
    target_chroms_norm = (
        {_normalize_chrom_name(ch) for ch in target_chroms_raw}
        if target_chroms_raw
        else None
    )
    fallback_name_count = 0

    def _strip_gene_version(gene_id: str) -> str:
        return gene_id.split(".", 1)[0] if "." in gene_id else gene_id

    def _is_ensembl_like(value: str) -> bool:
        return value.upper().startswith("ENS")

    def _pick_display_name(
        gene_name_attr: Optional[str],
        name_attr: Optional[str],
        gene_attr: Optional[str],
        fallback: str,
    ) -> str:
        for candidate in (gene_name_attr, gene_attr, name_attr):
            if candidate and not _is_ensembl_like(candidate):
                return candidate
        for candidate in (gene_name_attr, gene_attr, name_attr):
            if candidate:
                return candidate
        return fallback

    if gene_names:
        target_genes = set(gene_names)
        target_genes_norm = {_strip_gene_version(g) for g in gene_names}
        target_genes_lower = {g.lower() for g in gene_names}
        target_genes_lower_norm = {g.lower() for g in target_genes_norm}
        # Heuristic species hinting to catch human/mouse mismatches early.
        ensg_count = sum(1 for g in gene_names if g.upper().startswith("ENSG"))
        ensm_count = sum(1 for g in gene_names if g.upper().startswith("ENSMUSG"))
        mouse_like = sum(1 for g in gene_names if _is_mouse_like(g))
        total = max(len(gene_names), 1)
        stem = gtf_path.name.lower()
        gtf_human = any(
            token in stem
            for token in ("gencode.v", "grch", "hg", "gcf_000001405", "gcf_000001635")
        )
        gtf_mouse = any(
            token in stem for token in ("gencode.vm", "grcm", "mm", "gcf_000001635.")
        )
        if (ensm_count / total) > 0.2 or (mouse_like / total) > 0.2:
            if gtf_human and not gtf_mouse:
                _LOG.warning(
                    "Gene manifest looks mouse-like (ENSMUSG/Gm/Olfr/Rik). "
                    "GTF file name looks human: %s. This may cause missing genes.",
                    gtf_path.name,
                )
        if (ensg_count / total) > 0.2:
            if gtf_mouse and not gtf_human:
                _LOG.warning(
                    "Gene manifest looks human-like (ENSG). "
                    "GTF file name looks mouse: %s. This may cause missing genes.",
                    gtf_path.name,
                )
    else:
        target_genes = None
        target_genes_norm = set()
        target_genes_lower = set()
        target_genes_lower_norm = set()

    with opener(gtf_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 9:
                continue
            seqname, source, feature, start, end, score, strand, frame, attributes = (
                parts
            )
            if feature != "gene":
                continue
            chrom_norm = _normalize_chrom_name(seqname)
            if (
                target_chroms_raw
                and seqname not in target_chroms_raw
                and chrom_norm not in target_chroms_norm
            ):
                continue
            attr_map = _parse_gtf_attributes(attributes)
            gene_name_attr = attr_map.get("gene_name")
            name_attr = attr_map.get("Name")
            gene_attr = attr_map.get("gene")
            gene_id_raw = attr_map.get("gene_id") or attr_map.get("ID")
            if gene_id_raw is None:
                continue
            gene_id = _strip_gene_version(gene_id_raw)
            gene_name_raw = gene_name_attr or name_attr or gene_attr
            gene_name = _pick_display_name(
                gene_name_attr, name_attr, gene_attr, gene_id
            )
            if gene_name == gene_id and gene_name_raw is None:
                fallback_name_count += 1
            if target_genes:
                candidates = {
                    c for c in (gene_name_raw, gene_name, gene_id_raw, gene_id) if c
                }
                candidates_lower = {c.lower() for c in candidates}
                if not (
                    candidates & target_genes
                    or candidates & target_genes_norm
                    or candidates_lower & target_genes_lower
                    or candidates_lower & target_genes_lower_norm
                ):
                    continue
            start_i = int(start)
            end_i = int(end)
            strand_val = strand if strand in {"+", "-"} else "+"
            tss = start_i if strand_val == "+" else end_i
            records.append(
                GeneInfo(
                    gene_id=gene_id,
                    gene_name=gene_name,
                    chrom=chrom_norm,
                    tss=tss,
                    strand=strand_val,
                )
            )
    if target_genes and len(records) == 0:
        raise ValueError(
            "None of the requested genes were found in the provided GTF file"
        )
    if fallback_name_count:
        _LOG.warning(
            "GTF entries without gene names: %d (falling back to Ensembl IDs)",
            fallback_name_count,
        )
    _LOG.info("Parsed %d gene annotations", len(records))
    return records


def select_genes(
    genes: List[GeneInfo],
    requested_genes: Optional[Sequence[str]] = None,
    max_genes: Optional[int] = None,
) -> List[GeneInfo]:
    if requested_genes:
        request_set = set(requested_genes)
        request_lower = {g.lower() for g in requested_genes}
        filtered = [
            gene
            for gene in genes
            if gene.gene_name in request_set
            or gene.gene_id in request_set
            or gene.gene_name.lower() in request_lower
            or gene.gene_id.lower() in request_lower
        ]
        if not filtered:
            raise ValueError("Requested genes not found in annotations")
    else:
        filtered = genes

    # Always return a deterministic alphabetical ordering to keep training/inference aligned.
    filtered = sorted(filtered, key=lambda g: (g.gene_name.lower(), g.gene_id.lower()))
    if max_genes is not None:
        filtered = filtered[:max_genes]
    return filtered


def _resolve_gene_index(name_to_idx: Dict[str, int], gene: GeneInfo) -> Optional[int]:
    idx = name_to_idx.get(gene.gene_name)
    if idx is not None:
        return idx
    return name_to_idx.get(gene.gene_id)


def _gene_window_peak_indices(
    gene: GeneInfo,
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> np.ndarray:
    start = gene.tss - training.window_bp
    end = gene.tss + training.window_bp
    indices, midpoints = peak_indexer.get_peaks_in_window(gene.chrom, start, end)
    indices_ord, _ = _orient_peak_window(indices, midpoints, gene)
    return indices_ord


def _map_indices_to_union(idxs: np.ndarray, union_indices: np.ndarray) -> np.ndarray:
    """Map indices into their positions within a sorted union array.

    Returns an array of the same shape as ``idxs`` where missing entries are
    encoded as -1.
    """
    if idxs.size == 0:
        return np.array([], dtype=np.int64)
    positions = np.searchsorted(union_indices, idxs)
    valid = (positions < union_indices.size) & (union_indices[positions] == idxs)
    mapped = np.full(idxs.shape, -1, dtype=np.int64)
    mapped[valid] = positions[valid]
    return mapped


def _build_shared_peak_matrix(
    genes: List[GeneInfo],
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> tuple[np.ndarray | sp.spmatrix, List[str], List[np.ndarray]]:
    per_gene_peak_indices: List[np.ndarray] = []
    for gene in genes:
        peak_indices = _gene_window_peak_indices(gene, peak_indexer, training)
        per_gene_peak_indices.append(peak_indices)

    if not per_gene_peak_indices:
        raise ValueError("No peak windows constructed for shared peak matrix")

    union_peak_indices = np.unique(np.concatenate(per_gene_peak_indices))
    if union_peak_indices.size == 0:
        raise ValueError("No shared peaks found across selected genes")

    # Map global peak indices to positions within the union of peaks.
    # A value of -1 denotes peaks that are not present in the union set
    # (i.e., no corresponding shared feature).
    # Use a vectorized lookup based on the fact that ``union_peak_indices``
    # is sorted (as returned by ``np.unique``), avoiding per-element Python
    # loops for better performance when idxs is large.

    feature_block_indices = [
        _map_indices_to_union(idxs, union_peak_indices)
        for idxs in per_gene_peak_indices
    ]
    shared_matrix = peak_indexer.matrix[:, union_peak_indices]
    if sp.issparse(shared_matrix):
        shared_matrix = shared_matrix.tocsr()
    else:
        shared_matrix = np.asarray(shared_matrix, dtype=np.float32)

    feature_names = list(peak_indexer.peak_ids[union_peak_indices])
    return shared_matrix, feature_names, feature_block_indices


def _build_shared_bin_matrix(
    genes: List[GeneInfo],
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> tuple[np.ndarray, List[str], List[np.ndarray]]:
    bin_records: List[Tuple[str, int, int]] = []
    feature_names: List[str] = []
    feature_block_indices: List[np.ndarray] = []
    bin_map: Dict[Tuple[str, int, int], int] = {}

    for gene in genes:
        start = gene.tss - training.window_bp
        end = gene.tss + training.window_bp
        num_bins = int(np.ceil((end - start) / training.bin_size_bp))
        if num_bins <= 0:
            raise ValueError("Invalid bin configuration")
        bin_starts = start + np.arange(num_bins, dtype=np.int64) * training.bin_size_bp
        bin_ends = bin_starts + training.bin_size_bp

        indices: List[int] = []
        for bidx, (bstart, bend) in enumerate(zip(bin_starts, bin_ends)):
            record = (gene.chrom, int(bstart), int(bend))
            idx = bin_map.get(record)
            if idx is None:
                idx = len(bin_records)
                bin_records.append(record)
                feature_names.append(f"{record[0]}:{record[1]}-{record[2]}")
                bin_map[record] = idx
            indices.append(idx)

        if gene.strand == "-":
            indices = list(reversed(indices))
        feature_block_indices.append(np.asarray(indices, dtype=np.int64))

    if not bin_records:
        raise ValueError("No bins constructed for shared bin matrix")

    X = np.zeros((peak_indexer.n_cells, len(bin_records)), dtype=np.float32)
    for col_idx, (chrom, start, end) in enumerate(bin_records):
        indices, _ = peak_indexer.get_peaks_in_window(chrom, start, end)
        if indices.size == 0:
            continue
        matrix = peak_indexer.matrix[:, indices]
        if sp.issparse(matrix):
            matrix = matrix.tocsc()
            summed = np.asarray(matrix.sum(axis=1)).ravel()
        else:
            summed = np.sum(matrix, axis=1)
        X[:, col_idx] = summed

    return X, feature_names, feature_block_indices


def build_gene_dataset(
    gene: GeneInfo,
    atac: ad.AnnData,
    rna: ad.AnnData,
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> GeneDataset:
    rna_var = np.asarray(rna.var_names).astype(str)
    name_to_idx = {name: idx for idx, name in enumerate(rna_var)}
    gene_idx = _resolve_gene_index(name_to_idx, gene)
    if gene_idx is None:
        raise ValueError(f"Gene {gene.gene_name} not found in RNA matrix")

    if training.rna_expression_layer and training.rna_expression_layer in rna.layers:
        expression_matrix = rna.layers[training.rna_expression_layer]
        already_logged = "log1p" in training.rna_expression_layer.lower()
    else:
        expression_matrix = rna.X
        already_logged = False

    y = expression_matrix[:, gene_idx]
    if sp.issparse(y):
        y = np.asarray(y.toarray()).ravel()
    else:
        y = np.asarray(y).ravel()

    if training.log1p_transform and not already_logged:
        y = np.log1p(y)

    mask_source = rna.X[:, gene_idx]
    if sp.issparse(mask_source):
        mask_values = np.asarray(mask_source.toarray()).ravel()
    else:
        mask_values = np.asarray(mask_source).ravel()

    mask = mask_values >= training.min_expression
    cells_raw = int(mask_values.shape[0])
    cells_eligible = int(mask.sum())

    cell_ids = np.asarray(rna.obs_names).astype(str)
    if training.group_key and training.group_key in rna.obs.columns:
        group_labels = rna.obs[training.group_key].astype(str).to_numpy()
    else:
        if training.group_key:
            _LOG.warning(
                "group_key='%s' not found in RNA obs; falling back to ungrouped labels",
                training.group_key,
            )
        group_labels = np.asarray(rna.obs_names).astype(str)

    feature_basis = getattr(training, "per_gene_feature_basis", "bin")
    if feature_basis == "peak":
        features_all, feature_names = _compute_gene_peak_features_all_cells(
            gene, peak_indexer, training
        )
        peak_count = int(features_all.shape[1])
        min_peaks = int(getattr(training, "per_gene_peak_min_peaks", 1))
        if peak_count < min_peaks:
            raise ValueError(
                f"Gene {gene.gene_name} has only {peak_count} peaks in ±{training.window_bp} bp window; "
                f"requires at least {min_peaks} peaks for per-gene peak features"
            )
    else:
        features_all, feature_names = _compute_gene_features_all_cells(
            gene, peak_indexer, training
        )
        peak_count = None
    features_all, feature_names, _ = _append_global_features(
        features_all,
        feature_names,
        None,
        peak_indexer,
        training,
    )
    features = features_all.astype(np.float32)
    filter_mode = getattr(training, "per_gene_cell_filter_mode", "auto")
    apply_cell_filtering = filter_mode == "on"
    override_reason = str(getattr(training, "per_gene_cell_filter_reason", "") or "")
    if not apply_cell_filtering and not override_reason:
        override_reason = "explicit_off"

    min_cells_check = cells_eligible if apply_cell_filtering else cells_raw
    if min_cells_check < training.min_cells_per_gene:
        if apply_cell_filtering:
            raise ValueError(
                f"Gene {gene.gene_name} has only {cells_eligible} cells above expression threshold {training.min_expression}"
            )
        raise ValueError(
            f"Gene {gene.gene_name} has only {cells_raw} total cells available with cell filtering disabled"
        )

    if apply_cell_filtering:
        features = features[mask]
        y = y[mask]
        cell_ids = cell_ids[mask]
        group_labels = group_labels[mask]

    cells_filtered_out = cells_raw - (
        cells_eligible if apply_cell_filtering else cells_raw
    )
    metadata: Dict[str, object] = {
        "cells_raw": cells_raw,
        "cells_eligible": cells_eligible,
        "cells_filtered_out": cells_filtered_out,
        "feature_count": int(features.shape[1]),
        "feature_basis": feature_basis,
        "cell_filtering_applied": bool(apply_cell_filtering),
        "cell_filtering_override_reason": override_reason,
    }
    if peak_count is not None:
        metadata["peak_count_in_window"] = peak_count
        metadata["peak_window_bp"] = int(training.window_bp)
        metadata["peak_min_required"] = int(
            getattr(training, "per_gene_peak_min_peaks", 1)
        )
    return GeneDataset(
        gene=gene,
        X=features,
        y=y.astype(np.float32),
        cell_ids=cell_ids,
        feature_names=feature_names,
        group_labels=group_labels.astype(str),
        metadata=metadata,
    )


def _deduplicate_genes_by_rna_match(
    genes: List[GeneInfo],
    rna_var_names: np.ndarray,
) -> List[GeneInfo]:
    """
    Remove duplicate gene entries, prioritizing those whose gene_id is in the RNA data.

    When a GTF file has multiple entries with the same gene_name (e.g., IL3RA on chrX and chrY),
    this function keeps only the entry whose gene_id matches an entry in the RNA data.

    Tie-breaking for multiple RNA matches:
    - Prefer canonical autosomes (chr1-chr22) over sex/mito chromosomes
    - Within same chromosome category, sort by chromosome name lexicographically
    - Final tie-breaker: prefer upstream TSS (lower position on + strand, higher position on - strand)
    """
    from collections import defaultdict

    def _chromosome_sort_key(chrom: str) -> tuple:
        """Return sort key prioritizing canonical autosomes."""
        # Extract numeric part for sorting (chr1, chr2, ..., chr22)
        if chrom.startswith("chr"):
            suffix = chrom[3:]
            if suffix.isdigit():
                # Canonical autosome: sort by number
                return (0, int(suffix), chrom)
            elif suffix in {"X", "Y"}:
                # Sex chromosome: sort after autosomes
                return (1, 0, chrom)
            elif suffix in {"M", "MT"}:
                # Mitochondrial: sort last
                return (2, 0, chrom)
        # Unknown format: sort to end
        return (3, 0, chrom)

    rna_var_set = set(rna_var_names.astype(str))
    # Cache membership so duplicate gene_name groups do not repeatedly probe the RNA set
    gene_id_in_rna = {gene.gene_id: gene.gene_id in rna_var_set for gene in genes}
    name_to_genes = defaultdict(list)

    # Group genes by gene_name
    for gene in genes:
        name_to_genes[gene.gene_name].append(gene)

    deduplicated: List[GeneInfo] = []
    for gene_name, gene_list in name_to_genes.items():
        if len(gene_list) == 1:
            # No duplicates, keep as is
            deduplicated.append(gene_list[0])
        else:
            # Multiple entries with same gene_name - prioritize RNA matches
            matches_in_rna = [
                g for g in gene_list if gene_id_in_rna.get(g.gene_id, False)
            ]
            if matches_in_rna:
                # Sort multiple matches by chromosome (canonical first) then TSS position for determinism
                sorted_matches = sorted(
                    matches_in_rna,
                    key=lambda g: (
                        _chromosome_sort_key(g.chrom),
                        # Prefer upstream TSS: lower on + strand, higher on - strand via sign flip
                        g.tss if g.strand != "-" else -g.tss,
                    ),
                )
                selected = sorted_matches[0]
                deduplicated.append(selected)
                if len(matches_in_rna) > 1:
                    _LOG.warning(
                        "Gene %s has %d entries with gene_id in RNA data; selecting %s on %s (TSS=%d) over %s",
                        gene_name,
                        len(matches_in_rna),
                        selected.gene_id,
                        selected.chrom,
                        selected.tss,
                        ", ".join(
                            f"{m.gene_id} ({m.chrom})" for m in sorted_matches[1:]
                        ),
                    )
            else:
                # None match RNA data, keep first entry (will be skipped later)
                deduplicated.append(gene_list[0])
                _LOG.debug(
                    "Gene %s has %d GTF entries but none match RNA data; using %s",
                    gene_name,
                    len(gene_list),
                    gene_list[0].gene_id,
                )

    return deduplicated


def build_cellwise_dataset(
    genes: List[GeneInfo],
    atac: ad.AnnData,
    rna: ad.AnnData,
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> CellwiseDataset:
    if not genes:
        raise ValueError("No genes provided for cell-wise dataset construction")

    cell_ids = np.asarray(rna.obs_names).astype(str)
    rna_var = np.asarray(rna.var_names).astype(str)

    # Deduplicate genes, prioritizing those in RNA data
    genes = _deduplicate_genes_by_rna_match(genes, rna_var)

    name_to_idx = {name: idx for idx, name in enumerate(rna_var)}
    if training.group_key and training.group_key in rna.obs.columns:
        group_labels = rna.obs[training.group_key].astype(str).to_numpy()
    else:
        if training.group_key:
            _LOG.warning(
                "group_key='%s' not found in RNA obs; falling back to per-cell grouping",
                training.group_key,
            )
        group_labels = np.asarray(rna.obs_names).astype(str)

    if training.rna_expression_layer and training.rna_expression_layer in rna.layers:
        expression_matrix = rna.layers[training.rna_expression_layer]
        already_logged = "log1p" in training.rna_expression_layer.lower()
    else:
        expression_matrix = rna.X
        already_logged = False

    target_blocks: List[np.ndarray] = []
    raw_target_blocks: List[np.ndarray] = []
    selected_genes: List[GeneInfo] = []

    for gene in genes:
        gene_idx = _resolve_gene_index(name_to_idx, gene)
        if gene_idx is None:
            _LOG.warning("Gene %s not found in RNA matrix; skipping", gene.gene_name)
            continue

        y = expression_matrix[:, gene_idx]
        if sp.issparse(y):
            y = np.asarray(y.toarray()).ravel()
        else:
            y = np.asarray(y).ravel()

        if training.log1p_transform and not already_logged:
            y = np.log1p(y)

        raw_column = rna.X[:, gene_idx]
        if sp.issparse(raw_column):
            raw_values = np.asarray(raw_column.toarray()).ravel()
        else:
            raw_values = np.asarray(raw_column).ravel()

        if (raw_values >= training.min_expression).sum() < training.min_cells_per_gene:
            _LOG.warning(
                "Gene %s skipped due to insufficient expressing cells (%d < %d)",
                gene.gene_name,
                int((raw_values >= training.min_expression).sum()),
                training.min_cells_per_gene,
            )
            continue

        target_blocks.append(y.astype(np.float32))
        raw_target_blocks.append(raw_values.astype(np.float32))
        selected_genes.append(gene)

    if not selected_genes:
        raise ValueError("No genes satisfied inclusion criteria for cell-wise dataset")
    Y = _safe_stack(target_blocks, axis=1)
    raw_Y = _safe_stack(raw_target_blocks, axis=1)

    basis = training.multioutput_feature_basis
    if basis == "peak":
        shared_matrix, feature_names, feature_block_indices = _build_shared_peak_matrix(
            selected_genes,
            peak_indexer,
            training,
        )
    else:
        shared_matrix, feature_names, feature_block_indices = _build_shared_bin_matrix(
            selected_genes,
            peak_indexer,
            training,
        )
    shared_matrix, feature_names, feature_block_indices = _append_global_features(
        shared_matrix,
        feature_names,
        feature_block_indices,
        peak_indexer,
        training,
    )
    if feature_block_indices is None:
        feature_block_indices = []

    filter_mode = getattr(training, "per_gene_cell_filter_mode", "auto")
    apply_cell_filtering = filter_mode == "on"
    override_reason = str(getattr(training, "per_gene_cell_filter_reason", "") or "")
    if not apply_cell_filtering and not override_reason:
        override_reason = "explicit_off"

    cells_raw = int(cell_ids.shape[0])
    if apply_cell_filtering:
        cell_mask = np.any(raw_Y >= training.min_expression, axis=1)
        cells_eligible = int(cell_mask.sum())
        if cells_eligible < training.min_cells_per_gene:
            raise ValueError(
                "Cell-wise dataset has only "
                f"{cells_eligible} cells with any retained target >= {training.min_expression}"
            )
        shared_matrix = shared_matrix[cell_mask]
        Y = Y[cell_mask]
        cell_ids = cell_ids[cell_mask]
        group_labels = group_labels[cell_mask]
    else:
        cells_eligible = cells_raw

    cells_filtered_out = cells_raw - cells_eligible

    return CellwiseDataset(
        genes=selected_genes,
        X=shared_matrix,
        y=Y,
        cell_ids=cell_ids,
        feature_names=feature_names,
        group_labels=group_labels,
        metadata={
            "cells_raw": cells_raw,
            "cells_eligible": cells_eligible,
            "cells_filtered_out": cells_filtered_out,
            "cell_filtering_applied": bool(apply_cell_filtering),
            "cell_filtering_override_reason": override_reason,
            "feature_count": int(shared_matrix.shape[1]),
            "target_gene_count": len(selected_genes),
        },
        feature_block_slices=[],
        feature_block_indices=feature_block_indices,
    )


def build_cellwise_features_only(
    genes: List[GeneInfo],
    atac: ad.AnnData,
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> CellwiseDataset:
    """Construct cell-wise feature matrix without RNA targets (for inference)."""

    if not genes:
        raise ValueError("No genes provided for cell-wise feature construction")

    cell_ids = np.asarray(atac.obs_names).astype(str)
    group_labels = np.asarray(cell_ids)
    selected_genes: List[GeneInfo] = []

    for gene in genes:
        selected_genes.append(gene)

    if not selected_genes:
        raise ValueError("No genes produced usable feature blocks for inference")

    basis = training.multioutput_feature_basis
    if basis == "peak":
        shared_matrix, feature_names, feature_block_indices = _build_shared_peak_matrix(
            selected_genes,
            peak_indexer,
            training,
        )
    else:
        shared_matrix, feature_names, feature_block_indices = _build_shared_bin_matrix(
            selected_genes,
            peak_indexer,
            training,
        )
    shared_matrix, feature_names, feature_block_indices = _append_global_features(
        shared_matrix,
        feature_names,
        feature_block_indices,
        peak_indexer,
        training,
    )
    if feature_block_indices is None:
        feature_block_indices = []

    return CellwiseDataset(
        genes=selected_genes,
        X=shared_matrix,
        y=np.empty((shared_matrix.shape[0], 0), dtype=np.float32),
        cell_ids=cell_ids,
        feature_names=feature_names,
        group_labels=group_labels,
        feature_block_slices=[],
        feature_block_indices=feature_block_indices,
    )


def _safe_stack(arrays: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
    try:
        return np.stack(arrays, axis=axis)
    except MemoryError as exc:
        raise RuntimeError(
            "Cell-wise target stacking exceeded available memory. "
            "Consider reducing the number of genes or chunking jobs across nodes."
        ) from exc


def filter_atac_by_genes(
    atac: ad.AnnData, genes: Sequence[GeneInfo], window_bp: int
) -> ad.AnnData:
    """Subset the ATAC matrix to peaks located within +/- window_bp of the supplied genes' TSS."""

    if not genes:
        return atac

    var_df = atac.var.copy()
    chroms, starts, ends = _extract_peak_coordinates(var_df)
    if chroms.size == 0:
        _LOG.warning("ATAC peak metadata is empty; skipping coordinate-based filtering")
        return atac

    midpoints = (starts + ends) // 2
    keep_mask = np.zeros_like(midpoints, dtype=bool)

    chrom_to_indices: Dict[str, np.ndarray] = {}
    chrom_to_midpoints: Dict[str, np.ndarray] = {}
    for chrom in pd.unique(chroms):
        chrom_mask = chroms == chrom
        chrom_to_indices[chrom] = np.flatnonzero(chrom_mask)
        chrom_to_midpoints[chrom] = midpoints[chrom_mask]

    for gene in genes:
        idxs = chrom_to_indices.get(gene.chrom)
        if idxs is None or idxs.size == 0:
            continue
        mids = chrom_to_midpoints[gene.chrom]
        start = max(0, gene.tss - window_bp)
        end = gene.tss + window_bp
        local_mask = (mids >= start) & (mids <= end)
        if local_mask.any():
            keep_mask[idxs[local_mask]] = True

    selected_peaks = int(keep_mask.sum())
    if selected_peaks == 0:
        _LOG.warning(
            "Filtering by +/- %d bp around %d genes yielded no peaks; retaining original ATAC matrix",
            window_bp,
            len(genes),
        )
        return atac

    if selected_peaks == atac.shape[1]:
        _LOG.info(
            "All %d ATAC peaks already lie within +/- %d bp windows; no filtering applied",
            selected_peaks,
            window_bp,
        )
        return atac

    _LOG.info(
        "Filtered ATAC matrix from %d to %d peaks using +/- %d bp windows around %d genes",
        atac.shape[1],
        selected_peaks,
        window_bp,
        len(genes),
    )
    return atac[:, keep_mask].copy()


def _normalize_chrom_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return raw

    lowered = raw.lower()
    if lowered.startswith("chr"):
        if lowered in {"chrmt", "chrm"}:
            return "chrM"
        suffix = raw[3:]
        return f"chr{suffix}" if suffix else "chr"

    simple_alias = {"x": "chrX", "y": "chrY", "m": "chrM", "mt": "chrM"}
    if lowered in simple_alias:
        return simple_alias[lowered]

    if raw.isdigit():
        return f"chr{raw}"

    prefix = raw.split(".", 1)[0]
    mapped = _REFSEQ_TO_CHR.get(prefix)
    if mapped:
        return mapped

    return raw


def _is_mouse_like(name: str) -> bool:
    if not name:
        return False
    return bool(re.match(r"^(Gm\d+|[0-9]+Rik)$", name)) or name.startswith("Olfr")


def _parse_gtf_attributes(attributes: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in attributes.strip().split(";"):
        if not item.strip():
            continue
        entry = item.strip()
        if "=" in entry and " " not in entry:
            key, value = entry.split("=", 1)
        else:
            parts = entry.split(None, 1)
            if len(parts) == 2:
                key, value = parts
            else:
                key, value = parts[0], ""
        value = value.strip().strip('"')
        result[key] = value
    return result


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
        upper = name.upper()
        lower = name.lower()
        title = name.title()
        if upper in df.columns:
            return upper
        if lower in df.columns:
            return lower
        if title in df.columns:
            return title
    raise ValueError(
        f"None of the candidate columns {candidates} were found in dataframe"
    )


def _parse_peak_index(
    index_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    chroms: List[str] = []
    starts: List[int] = []
    ends: List[int] = []

    for entry in index_values:
        text = str(entry)
        if not text:
            raise ValueError("Encountered empty peak coordinate in ATAC index")
        try:
            chrom_part, range_part = text.split(":", 1)
            start_part, end_part = range_part.split("-", 1)
            chroms.append(chrom_part)
            starts.append(int(start_part))
            ends.append(int(end_part))
        except ValueError as exc:
            raise ValueError(
                f"Unable to parse peak coordinate '{text}'. Expected format 'chr:start-end'."
            ) from exc

    if not chroms:
        raise ValueError("No peak coordinates could be parsed from ATAC index")

    return (
        np.asarray(chroms, dtype=str),
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
    )


def _extract_peak_coordinates(
    var_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        chrom_col = _find_column(var_df, ["chrom", "chr", "chromosome", "seqname"])
        start_col = _find_column(var_df, ["start", "chromStart", "begin", "sta"])
        end_col = _find_column(var_df, ["end", "chromEnd", "stop", "sto"])
        chroms = var_df[chrom_col].astype(str).to_numpy()
        starts = var_df[start_col].astype(np.int64).to_numpy()
        ends = var_df[end_col].astype(np.int64).to_numpy()
    except ValueError:
        _LOG.info("ATAC peaks missing explicit coordinate columns; parsing from index")
        chroms, starts, ends = _parse_peak_index(np.asarray(var_df.index).astype(str))
    return chroms, starts, ends

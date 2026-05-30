#!/usr/bin/env python3
"""Preprocess GEO raw multiome inputs into SPEAR-ready AnnData files."""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import pysam
import scanpy as sc
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.conversion import localconverter

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spear.preprocessing import align_modalities  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOG = logging.getLogger(__name__)
ad.settings.allow_write_nullable_strings = True


EMBRYONIC_SAMPLE_ORDER = [
    "E7.5_rep1",
    "E7.5_rep2",
    "E7.75_rep1",
    "E8.0_rep1",
    "E8.0_rep2",
    "E8.5_CRISPR_T_KO",
    "E8.5_CRISPR_T_WT",
    "E8.5_rep1",
    "E8.5_rep2",
    "E8.75_rep1",
    "E8.75_rep2",
]

ENDOTHELIAL_SAMPLES = {
    "S3H_hypoxia": {"condition": "hypoxia", "source_gsm": "GSM8335427"},
    "S3N_normoxia": {"condition": "normoxia", "source_gsm": "GSM8335429"},
}

EMBRYONIC_MT_THRESHOLD = 15.0
ENDOTHELIAL_MT_THRESHOLD = 15.0


def normalize_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    for axis_name in ("obs", "var"):
        frame = getattr(adata, axis_name)
        frame.index = pd.Index(frame.index.astype(str), dtype=object)
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_string_dtype(series.dtype):
                frame[column] = series.astype(object)
    return adata


def detect_10x_prefix(sample_dir: Path) -> str:
    features = sorted(sample_dir.glob("*features.tsv.gz"))
    if len(features) != 1:
        raise FileNotFoundError(
            f"Expected exactly one features.tsv.gz in {sample_dir}, found {features}"
        )
    return features[0].name.replace("features.tsv.gz", "")


def load_rna_10x(sample_dir: Path) -> ad.AnnData:
    prefix = detect_10x_prefix(sample_dir)
    rna = sc.read_10x_mtx(
        sample_dir.as_posix(),
        prefix=prefix,
        var_names="gene_symbols",
        make_unique=True,
    )
    if "gene_ids" not in rna.var.columns:
        first_col = rna.var.columns[0] if len(rna.var.columns) else None
        if first_col is not None:
            rna.var["gene_ids"] = rna.var[first_col].astype(str)
        else:
            rna.var["gene_ids"] = rna.var_names.astype(str)
    rna.obs["barcode"] = pd.Index(rna.obs_names).astype(str)
    return rna


def load_embryonic_peaks(raw_dir: Path) -> pd.DataFrame:
    peaks_path = raw_dir.parents[1] / "references" / "GSE205117_ATAC_peaks.tsv.gz"
    if not peaks_path.exists():
        raise FileNotFoundError(f"Missing embryonic peaks reference: {peaks_path}")
    peaks = pd.read_csv(peaks_path, sep="\t")
    peaks = peaks.rename(columns={"chr": "chromosome"})
    peaks["peak"] = (
        peaks["chromosome"].astype(str)
        + ":"
        + peaks["start"].astype(int).astype(str)
        + "-"
        + peaks["end"].astype(int).astype(str)
    )
    return peaks[["peak", "chromosome", "start", "end"]].copy()


def load_endothelial_peaks(raw_dir: Path) -> pd.DataFrame:
    peaks_gz_path = raw_dir / "GSE270141_peaks_MACS2.rds.gz"
    temp_rds = raw_dir / "GSE270141_peaks_MACS2.rds"
    if not peaks_gz_path.exists() and not temp_rds.exists():
        raise FileNotFoundError(
            "Missing endothelial peaks reference: expected "
            f"{peaks_gz_path} or {temp_rds}"
        )

    if not temp_rds.exists():
        LOG.info("Decompressing %s", peaks_gz_path.name)
        with gzip.open(peaks_gz_path, "rb") as src, temp_rds.open("wb") as dst:
            dst.write(src.read())

    r(f"""
        suppressPackageStartupMessages(library(GenomicRanges))
        .spear_peaks_obj <- readRDS("{temp_rds.as_posix()}")
        .spear_peaks_df <- data.frame(
            chromosome = as.character(seqnames(.spear_peaks_obj)),
            start = start(.spear_peaks_obj),
            end = end(.spear_peaks_obj)
        )
        """)
    with localconverter(pandas2ri.converter):
        peaks = r(".spear_peaks_df")
    peaks["peak"] = (
        peaks["chromosome"].astype(str)
        + ":"
        + peaks["start"].astype(int).astype(str)
        + "-"
        + peaks["end"].astype(int).astype(str)
    )
    return peaks[["peak", "chromosome", "start", "end"]].copy()


def ensure_tabix_index(fragments_path: Path) -> Path:
    index_path = Path(f"{fragments_path}.tbi")
    if index_path.exists():
        return index_path
    LOG.info("Indexing fragments: %s", fragments_path)
    pysam.tabix_index(
        fragments_path.as_posix(),
        preset="bed",
        force=True,
        keep_original=True,
    )
    if not index_path.exists():
        raise FileNotFoundError(f"Failed to create tabix index for {fragments_path}")
    return index_path


def build_atac_from_fragments(
    *,
    fragments_path: Path,
    allowed_barcodes: Iterable[str],
    peaks: pd.DataFrame,
) -> ad.AnnData:
    from muon import atac as ac

    ensure_tabix_index(fragments_path)
    obs = pd.DataFrame(index=pd.Index(sorted(set(allowed_barcodes)), dtype="string"))
    obs["barcode"] = obs.index.astype(str)
    shell = ad.AnnData(obs=obs)
    ac.tl.locate_fragments(shell, fragments_path.as_posix())
    atac = ac.tl.count_fragments_features(
        shell,
        features=peaks,
        stranded=False,
        extend_upstream=0,
        extend_downstream=0,
        count_reads=False,
    )
    if "barcode" in atac.obs.columns:
        atac.obs_names = pd.Index(atac.obs["barcode"].astype(str))
    atac.var_names = peaks["peak"].astype(str)
    atac.var["gene_ids"] = atac.var_names.astype(str)
    return atac


def annotate_sample_prefix(
    rna: ad.AnnData,
    atac: ad.AnnData,
    *,
    sample_name: str,
    obs_updates: dict[str, object] | None = None,
) -> tuple[ad.AnnData, ad.AnnData]:
    rna, atac = align_modalities(rna, atac)
    obs_updates = obs_updates or {}

    raw_barcodes = pd.Index(rna.obs_names).astype(str)
    prefixed = pd.Index(
        [f"{sample_name}.{barcode}" for barcode in raw_barcodes], dtype="string"
    )
    rna.obs_names = prefixed
    atac.obs_names = prefixed

    for adata in (rna, atac):
        adata.obs["barcode"] = raw_barcodes.astype(str).values
        adata.obs["sample"] = sample_name
        for key, value in obs_updates.items():
            adata.obs[key] = value
    return rna, atac


def read_barcode_set(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing barcode file: {path}")
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def subset_modalities_by_barcodes(
    rna: ad.AnnData,
    atac: ad.AnnData,
    *,
    barcodes: Iterable[str],
) -> tuple[ad.AnnData, ad.AnnData]:
    barcode_set = set(map(str, barcodes))
    keep = pd.Index(
        [barcode for barcode in map(str, rna.obs_names) if barcode in barcode_set],
        dtype="string",
    )
    if keep.empty:
        raise ValueError(
            "No requested barcodes were found in the pooled endothelial subset"
        )
    return rna[keep].copy(), atac[keep].copy()


def load_pooled_endothelial_modalities(raw_dir: Path) -> tuple[ad.AnnData, ad.AnnData]:
    rna_path = raw_dir / "endo_rna_only.h5ad"
    atac_path = raw_dir / "endo_atac_only.h5ad"
    if not rna_path.exists() or not atac_path.exists():
        raise FileNotFoundError(
            "Missing pooled endothelial inputs: expected " f"{rna_path} and {atac_path}"
        )
    LOG.info(
        "Loading pooled endothelial subset from %s and %s",
        rna_path.name,
        atac_path.name,
    )
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)
    return align_modalities(rna, atac)


def preprocess_endothelial_from_pooled(
    base_dir: Path,
    *,
    samples: Sequence[str] | None = None,
    combine: bool = False,
) -> tuple[Path, Path] | None:
    raw_dir = base_dir / "endothelial" / "raw"
    processed_root = base_dir / "endothelial" / "processed"
    tss_gene_df = _build_tss_gene_df(
        base_dir / "references" / "gencode.v44.annotation.gtf.gz"
    )
    load_endothelial_peaks(raw_dir)
    pooled_rna, pooled_atac = load_pooled_endothelial_modalities(raw_dir)
    pooled_barcodes = set(map(str, pooled_rna.obs_names))

    hypoxia_barcodes = read_barcode_set(raw_dir / "GSM8335427_S3H_GEX_barcodes.tsv")
    normoxia_barcodes = read_barcode_set(raw_dir / "GSM8335429_S3N_GEX_barcodes.tsv")
    overlap_barcodes = (hypoxia_barcodes & normoxia_barcodes) & pooled_barcodes
    if overlap_barcodes:
        LOG.info(
            "Dropping %d endothelial barcodes present in both hypoxia and normoxia lists",
            len(overlap_barcodes),
        )

    sample_to_barcodes = {
        "S3H_hypoxia": (hypoxia_barcodes - overlap_barcodes) & pooled_barcodes,
        "S3N_normoxia": (normoxia_barcodes - overlap_barcodes) & pooled_barcodes,
    }

    rna_samples: list[ad.AnnData] = []
    atac_samples: list[ad.AnnData] = []
    sample_order = list(samples) if samples else list(ENDOTHELIAL_SAMPLES.keys())

    for sample_name in sample_order:
        metadata = ENDOTHELIAL_SAMPLES[sample_name]

        def builder(
            sample_name: str = sample_name,
            metadata: dict[str, str] = metadata,
            tss_gene_df: pd.DataFrame | None = tss_gene_df,
        ):
            LOG.info("Processing pooled endothelial sample %s", sample_name)
            sample_rna, sample_atac = subset_modalities_by_barcodes(
                pooled_rna,
                pooled_atac,
                barcodes=sample_to_barcodes[sample_name],
            )
            sample_rna, sample_atac = qc_pair(
                sample_rna,
                sample_atac,
                species="human",
                mt_threshold=ENDOTHELIAL_MT_THRESHOLD,
                tss_gene_df=tss_gene_df,
            )
            sample_rna, sample_atac = annotate_sample_prefix(
                sample_rna,
                sample_atac,
                sample_name=sample_name,
                obs_updates=metadata,
            )
            for adata in (sample_rna, sample_atac):
                adata.obs["oxygen"] = metadata["condition"]
            return sample_rna, sample_atac

        rna, atac = load_or_process_sample(
            processed_root=processed_root,
            sample_name=sample_name,
            builder=builder,
        )
        rna_samples.append(rna)
        atac_samples.append(atac)

    if not combine:
        return None

    return write_outputs(
        dataset_dir=base_dir / "endothelial",
        rna_samples=rna_samples,
        atac_samples=atac_samples,
        dataset_name="endothelial",
    )


def _build_tss_gene_df(gtf_path: Path) -> pd.DataFrame | None:
    try:
        from spear.data import parse_gtf

        genes = parse_gtf(gtf_path)
    except Exception as exc:
        LOG.warning("Could not parse GTF for TSS enrichment: %s", exc)
        return None
    seen: set[str] = set()
    rows = []
    for gene in genes:
        if gene.gene_name in seen:
            continue
        seen.add(gene.gene_name)
        rows.append(
            {
                "gene_name": gene.gene_name,
                "Chromosome": gene.chrom,
                "Start": gene.tss,
                "End": gene.tss + 1,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("gene_name")


def mitochondrial_mask(rna: ad.AnnData, species: str) -> pd.Series:
    gene_names = pd.Index(rna.var_names).astype(str)
    prefix = "MT-" if species == "human" else "mt-"
    return pd.Series(
        gene_names.str.upper().str.startswith(prefix.upper()), index=rna.var_names
    )


def qc_pair(
    rna: ad.AnnData,
    atac: ad.AnnData,
    *,
    species: str,
    mt_threshold: float,
    min_rna_genes: int = 200,
    max_rna_genes: int = 7500,
    min_rna_cells: int = 3,
    min_atac_features: int = 200,
    min_atac_cells: int = 10,
    max_atac_total_counts: int = 100_000,
    ns_threshold: float = 4.0,
    tss_threshold: float = 2.0,
    fragments_path: Path | None = None,
    tss_gene_df: pd.DataFrame | None = None,
) -> tuple[ad.AnnData, ad.AnnData]:
    rna, atac = align_modalities(rna, atac)

    # RNA QC: lower bounds + MT filter + upper bound to remove likely doublets
    rna.var["mt"] = mitochondrial_mask(rna, species)
    sc.pp.calculate_qc_metrics(
        rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    sc.pp.filter_cells(rna, min_genes=min_rna_genes)
    sc.pp.filter_cells(rna, max_genes=max_rna_genes)
    sc.pp.filter_genes(rna, min_cells=min_rna_cells)
    if "pct_counts_mt" in rna.obs:
        rna = rna[rna.obs["pct_counts_mt"] < mt_threshold].copy()

    # ATAC QC: lower bounds + upper count bound to remove likely doublets
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(atac, min_genes=min_atac_features)
    sc.pp.filter_cells(atac, max_counts=max_atac_total_counts)
    sc.pp.filter_genes(atac, min_cells=min_atac_cells)

    # Fragment-based ATAC QC: nucleosome signal + TSS enrichment
    if fragments_path is not None:
        from muon import atac as ac

        try:
            ac.tl.locate_fragments(atac, fragments_path.as_posix())
            ac.tl.nucleosome_signal(atac, n=1_000_000)
            if "nucleosome_signal" in atac.obs:
                n_before = atac.n_obs
                atac = atac[atac.obs["nucleosome_signal"] <= ns_threshold].copy()
                LOG.info(
                    "Nucleosome signal filter: %d → %d cells", n_before, atac.n_obs
                )
        except Exception as exc:
            LOG.warning("Nucleosome signal QC skipped: %s", exc)

        if tss_gene_df is not None:
            try:
                ac.tl.tss_enrichment(atac, features=tss_gene_df, n_tss=1000)
                if "tss_score" in atac.obs:
                    scores = atac.obs["tss_score"]
                    would_keep = (scores >= tss_threshold).sum()
                    retention = would_keep / len(scores)
                    LOG.info(
                        "TSS enrichment scores computed: median=%.2f, would retain %d/%d cells at threshold %.1f (%.0f%%)",
                        scores.median(),
                        would_keep,
                        len(scores),
                        tss_threshold,
                        retention * 100,
                    )
                    if retention >= 0.50:
                        atac = atac[scores >= tss_threshold].copy()
                        LOG.info(
                            "TSS enrichment filter applied: %d cells retained",
                            atac.n_obs,
                        )
                    else:
                        LOG.warning(
                            "TSS enrichment filter skipped — would drop >50%% of cells "
                            "(retention %.0f%% < 50%%). Scores stored in obs['tss_score'] for inspection.",
                            retention * 100,
                        )
            except Exception as exc:
                LOG.warning("TSS enrichment QC skipped: %s", exc)

    rna, atac = align_modalities(rna, atac)
    return rna, atac


def _sample_from_obs_names(obs_names: Sequence[str]) -> pd.Series:
    values = [
        str(n).split(".", 1)[0] if "." in str(n) else "unknown" for n in obs_names
    ]
    return pd.Series(values, index=pd.Index(obs_names), dtype="string")


def ensure_barcode_and_sample(
    adata: ad.AnnData, sample_key: str = "sample"
) -> ad.AnnData:
    if "barcode" not in adata.obs:
        adata.obs["barcode"] = pd.Index(adata.obs_names).astype(str)
    if sample_key not in adata.obs:
        adata.obs[sample_key] = _sample_from_obs_names(adata.obs_names)
    adata.obs[sample_key] = adata.obs[sample_key].astype("string")
    return adata


def annotate_embryonic_metadata(
    rna: ad.AnnData,
    atac: ad.AnnData,
    sample_key: str = "sample",
) -> tuple[ad.AnnData, ad.AnnData]:
    rna, atac = align_modalities(
        ensure_barcode_and_sample(rna, sample_key=sample_key),
        ensure_barcode_and_sample(atac, sample_key=sample_key),
    )
    sample_series = rna.obs[sample_key].astype("string")
    metadata = pd.DataFrame(index=rna.obs_names)
    metadata["sample"] = sample_series
    metadata["stage"] = sample_series.str.extract(
        r"^(E\d+(?:\.\d+)?)", expand=False
    ).astype("string")
    metadata["replicate"] = sample_series.str.extract(r"(rep\d+)", expand=False).astype(
        "string"
    )
    perturbation = pd.Series("timecourse", index=metadata.index, dtype="string")
    perturbation = perturbation.mask(
        sample_series.str.contains("CRISPR_T_KO", na=False), "CRISPR_T_KO"
    )
    perturbation = perturbation.mask(
        sample_series.str.contains("CRISPR_T_WT", na=False), "CRISPR_T_WT"
    )
    metadata["perturbation"] = perturbation
    metadata["condition"] = np.where(
        metadata["perturbation"].eq("timecourse"),
        "wildtype_timecourse",
        metadata["perturbation"],
    )
    metadata["condition"] = metadata["condition"].astype("string")
    metadata["timepoint"] = metadata["stage"].astype("string")
    metadata["timepoint_numeric"] = pd.to_numeric(
        metadata["stage"].str.removeprefix("E"), errors="coerce"
    )
    metadata["is_timecourse"] = metadata["perturbation"].eq("timecourse")
    for adata in (rna, atac):
        for column in metadata.columns:
            adata.obs[column] = metadata[column]
    return rna, atac


def write_outputs(
    *,
    dataset_dir: Path,
    rna_samples: list[ad.AnnData],
    atac_samples: list[ad.AnnData],
    dataset_name: str,
    prefix: str = "combined",
) -> tuple[Path, Path]:
    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    combined_rna = ad.concat(rna_samples, join="outer", merge="same")
    combined_atac = ad.concat(atac_samples, join="outer", merge="same")
    combined_rna, combined_atac = align_modalities(combined_rna, combined_atac)

    if dataset_name == "embryonic":
        combined_rna, combined_atac = annotate_embryonic_metadata(
            combined_rna, combined_atac
        )

    rna_path = processed_dir / f"{prefix}_RNA_muon_raw_qc.h5ad"
    atac_path = processed_dir / f"{prefix}_ATAC_muon_raw_qc.h5ad"
    normalize_for_h5ad(combined_rna).write_h5ad(rna_path)
    normalize_for_h5ad(combined_atac).write_h5ad(atac_path)
    LOG.info("Wrote %s", rna_path)
    LOG.info("Wrote %s", atac_path)
    return rna_path, atac_path


def load_or_process_sample(
    *,
    processed_root: Path,
    sample_name: str,
    builder,
) -> tuple[ad.AnnData, ad.AnnData]:
    sample_dir = processed_root / "per_sample" / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    rna_path = sample_dir / f"{sample_name}_RNA_qc.h5ad"
    atac_path = sample_dir / f"{sample_name}_ATAC_qc.h5ad"

    if rna_path.exists() and atac_path.exists():
        LOG.info("Reusing cached sample %s", sample_name)
        return ad.read_h5ad(rna_path), ad.read_h5ad(atac_path)

    rna, atac = builder()
    normalize_for_h5ad(rna).write_h5ad(rna_path)
    normalize_for_h5ad(atac).write_h5ad(atac_path)
    LOG.info("Cached sample %s", sample_name)
    return rna, atac


def preprocess_embryonic(
    base_dir: Path,
    *,
    samples: Sequence[str] | None = None,
    combine: bool = False,
) -> tuple[Path, Path] | None:
    raw_dir = base_dir / "embryonic" / "raw"
    processed_root = base_dir / "embryonic" / "processed"
    peaks = load_embryonic_peaks(raw_dir)
    tss_gene_df = _build_tss_gene_df(
        base_dir / "references" / "GCF_000001635.27_genomic.gtf"
    )
    rna_samples: list[ad.AnnData] = []
    atac_samples: list[ad.AnnData] = []
    sample_order = list(samples) if samples else list(EMBRYONIC_SAMPLE_ORDER)

    for sample_name in sample_order:

        def builder(
            sample_name: str = sample_name,
            tss_gene_df: pd.DataFrame | None = tss_gene_df,
        ):
            sample_dir = raw_dir / sample_name
            fragments_files = list(sample_dir.glob("*ATAC_fragments.tsv.gz"))
            fragments = fragments_files[0] if fragments_files else None
            LOG.info("Processing embryonic sample %s", sample_name)
            rna = load_rna_10x(sample_dir)
            atac = build_atac_from_fragments(
                fragments_path=fragments,
                allowed_barcodes=rna.obs_names,
                peaks=peaks,
            )
            rna, atac = qc_pair(
                rna,
                atac,
                species="mouse",
                mt_threshold=EMBRYONIC_MT_THRESHOLD,
                fragments_path=fragments,
                tss_gene_df=tss_gene_df,
            )
            return annotate_sample_prefix(rna, atac, sample_name=sample_name)

        rna, atac = load_or_process_sample(
            processed_root=processed_root,
            sample_name=sample_name,
            builder=builder,
        )
        rna_samples.append(rna)
        atac_samples.append(atac)

    if not combine:
        return None

    return write_outputs(
        dataset_dir=base_dir / "embryonic",
        rna_samples=rna_samples,
        atac_samples=atac_samples,
        dataset_name="embryonic",
    )


def preprocess_endothelial(
    base_dir: Path,
    *,
    samples: Sequence[str] | None = None,
    combine: bool = False,
) -> tuple[Path, Path] | None:
    raw_dir = base_dir / "endothelial" / "raw"
    pooled_rna_path = raw_dir / "endo_rna_only.h5ad"
    pooled_atac_path = raw_dir / "endo_atac_only.h5ad"
    if pooled_rna_path.exists() and pooled_atac_path.exists():
        return preprocess_endothelial_from_pooled(
            base_dir, samples=samples, combine=combine
        )

    processed_root = base_dir / "endothelial" / "processed"
    peaks = load_endothelial_peaks(raw_dir)
    tss_gene_df = _build_tss_gene_df(
        base_dir / "references" / "gencode.v44.annotation.gtf.gz"
    )
    rna_samples: list[ad.AnnData] = []
    atac_samples: list[ad.AnnData] = []
    sample_order = list(samples) if samples else list(ENDOTHELIAL_SAMPLES.keys())

    for sample_name in sample_order:
        metadata = ENDOTHELIAL_SAMPLES[sample_name]

        def builder(
            sample_name: str = sample_name,
            metadata: dict[str, str] = metadata,
            tss_gene_df: pd.DataFrame | None = tss_gene_df,
        ):
            sample_dir = raw_dir / sample_name
            fragments_files = list(sample_dir.glob("*ATAC_fragments.tsv.gz"))
            fragments = fragments_files[0] if fragments_files else None
            LOG.info("Processing endothelial sample %s", sample_name)
            rna = load_rna_10x(sample_dir)
            atac = build_atac_from_fragments(
                fragments_path=fragments,
                allowed_barcodes=rna.obs_names,
                peaks=peaks,
            )
            rna, atac = qc_pair(
                rna,
                atac,
                species="human",
                mt_threshold=ENDOTHELIAL_MT_THRESHOLD,
                fragments_path=fragments,
                tss_gene_df=tss_gene_df,
            )
            rna, atac = annotate_sample_prefix(
                rna, atac, sample_name=sample_name, obs_updates=metadata
            )
            for adata in (rna, atac):
                adata.obs["oxygen"] = metadata["condition"]
            return rna, atac

        rna, atac = load_or_process_sample(
            processed_root=processed_root,
            sample_name=sample_name,
            builder=builder,
        )
        rna_samples.append(rna)
        atac_samples.append(atac)

    if not combine:
        return None

    return write_outputs(
        dataset_dir=base_dir / "endothelial",
        rna_samples=rna_samples,
        atac_samples=atac_samples,
        dataset_name="endothelial",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess GEO raw multiome data into SPEAR-ready h5ad files"
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path.cwd() / "data"),
        help="Base directory containing embryonic/ and endothelial/ subdirectories (default: ./data)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("embryonic", "endothelial", "all"),
        default=["all"],
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        help="Optional sample names to preprocess. Defaults to all samples in the selected dataset.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also write dataset-level combined h5ad files. Disabled by default.",
    )
    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir).expanduser().resolve()
    requested = set(args.datasets)
    if "all" in requested:
        requested = {"embryonic", "endothelial"}

    if "embryonic" in requested:
        preprocess_embryonic(base_dir, samples=args.samples, combine=args.combine)
    if "endothelial" in requested:
        preprocess_endothelial(base_dir, samples=args.samples, combine=args.combine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

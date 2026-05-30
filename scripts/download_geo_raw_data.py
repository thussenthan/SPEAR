#!/usr/bin/env python3
"""Stage or download GEO raw inputs into data/."""

from __future__ import annotations

import argparse
import gzip
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

EMBRYONIC_SAMPLES: dict[str, list[str]] = {
    "E7.5_rep1": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205416/"
        "suppl/GSM6205416%5FE7.5%5Frep1%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205416/"
        "suppl/GSM6205416%5FE7.5%5Frep1%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205416/"
        "suppl/GSM6205416%5FE7.5%5Frep1%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205427/"
        "suppl/GSM6205427%5FE7.5%5Frep1%5FATAC%5Ffragments.tsv.gz",
    ],
    "E7.5_rep2": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205417/"
        "suppl/GSM6205417%5FE7.5%5Frep2%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205417/"
        "suppl/GSM6205417%5FE7.5%5Frep2%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205417/"
        "suppl/GSM6205417%5FE7.5%5Frep2%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205428/"
        "suppl/GSM6205428%5FE7.5%5Frep2%5FATAC%5Ffragments.tsv.gz",
    ],
    "E7.75_rep1": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205418/"
        "suppl/GSM6205418%5FE7.75%5Frep1%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205418/"
        "suppl/GSM6205418%5FE7.75%5Frep1%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205418/"
        "suppl/GSM6205418%5FE7.75%5Frep1%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205429/"
        "suppl/GSM6205429%5FE7.75%5Frep1%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.0_rep1": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205419/"
        "suppl/GSM6205419%5FE8.0%5Frep1%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205419/"
        "suppl/GSM6205419%5FE8.0%5Frep1%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205419/"
        "suppl/GSM6205419%5FE8.0%5Frep1%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205430/"
        "suppl/GSM6205430%5FE8.0%5Frep1%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.0_rep2": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205420/"
        "suppl/GSM6205420%5FE8.0%5Frep2%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205420/"
        "suppl/GSM6205420%5FE8.0%5Frep2%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205420/"
        "suppl/GSM6205420%5FE8.0%5Frep2%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205431/"
        "suppl/GSM6205431%5FE8.0%5Frep2%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.5_CRISPR_T_KO": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205421/"
        "suppl/GSM6205421%5FE8.5%5FCRISPR%5FT%5FKO%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205421/"
        "suppl/GSM6205421%5FE8.5%5FCRISPR%5FT%5FKO%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205421/"
        "suppl/GSM6205421%5FE8.5%5FCRISPR%5FT%5FKO%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205432/"
        "suppl/GSM6205432%5FE8.5%5FCRISPR%5FT%5FKO%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.5_CRISPR_T_WT": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205422/"
        "suppl/GSM6205422%5FE8.5%5FCRISPR%5FT%5FWT%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205422/"
        "suppl/GSM6205422%5FE8.5%5FCRISPR%5FT%5FWT%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205422/"
        "suppl/GSM6205422%5FE8.5%5FCRISPR%5FT%5FWT%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205433/"
        "suppl/GSM6205433%5FE8.5%5FCRISPR%5FT%5FWT%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.5_rep1": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205423/"
        "suppl/GSM6205423%5FE8.5%5Frep1%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205423/"
        "suppl/GSM6205423%5FE8.5%5Frep1%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205423/"
        "suppl/GSM6205423%5FE8.5%5Frep1%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205434/"
        "suppl/GSM6205434%5FE8.5%5Frep1%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.5_rep2": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205424/"
        "suppl/GSM6205424%5FE8.5%5Frep2%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205424/"
        "suppl/GSM6205424%5FE8.5%5Frep2%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205424/"
        "suppl/GSM6205424%5FE8.5%5Frep2%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205435/"
        "suppl/GSM6205435%5FE8.5%5Frep2%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.75_rep1": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205425/"
        "suppl/GSM6205425%5FE8.75%5Frep1%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205425/"
        "suppl/GSM6205425%5FE8.75%5Frep1%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205425/"
        "suppl/GSM6205425%5FE8.75%5Frep1%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205436/"
        "suppl/GSM6205436%5FE8.75%5Frep1%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.75_rep2": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205426/"
        "suppl/GSM6205426%5FE8.75%5Frep2%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205426/"
        "suppl/GSM6205426%5FE8.75%5Frep2%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205426/"
        "suppl/GSM6205426%5FE8.75%5Frep2%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205437/"
        "suppl/GSM6205437%5FE8.75%5Frep2%5FATAC%5Ffragments.tsv.gz",
    ],
}

ENDOTHELIAL_SAMPLES: dict[str, list[str]] = {
    "S3H_hypoxia": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8335nnn/GSM8335427/"
        "suppl/GSM8335427%5FS3H%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8335nnn/GSM8335427/"
        "suppl/GSM8335427%5FS3H%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8335nnn/GSM8335427/"
        "suppl/GSM8335427%5FS3H%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8335nnn/GSM8335428/"
        "suppl/GSM8335428%5FS3H%5FATAC%5Ffragments.tsv.gz",
    ],
    "S3N_normoxia": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8335nnn/GSM8335429/"
        "suppl/GSM8335429%5FS3N%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8335nnn/GSM8335429/"
        "suppl/GSM8335429%5FS3N%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8335nnn/GSM8335429/"
        "suppl/GSM8335429%5FS3N%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8335nnn/GSM8335430/"
        "suppl/GSM8335430%5FS3N%5FATAC%5Ffragments.tsv.gz",
    ],
}

ENDOTHELIAL_SUPPLEMENTARY_URLS: dict[str, str] = {
    "GSE270141_peaks_MACS2.rds.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE270nnn/GSE270141/"
        "suppl/GSE270141_peaks_MACS2.rds.gz"
    ),
}

ENDOTHELIAL_SERIES_ARCHIVE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE270nnn/GSE270141/suppl/GSE270141_RAW.tar"

ENDOTHELIAL_DERIVED_CACHE_FILES: tuple[str, ...] = (
    "combined_barcodes.tsv",
    "GSM8335427_S3H_GEX_barcodes.tsv",
    "GSM8335429_S3N_GEX_barcodes.tsv",
    "sce_rna.rds",
    "sce_atac.rds",
    "endo_rna_only.h5ad",
    "endo_atac_only.h5ad",
)

ENDOTHELIAL_DERIVED_CACHE_DIRS: tuple[str, ...] = ("fragments",)

REFERENCE_URLS: dict[str, str] = {
    "GSE205117_ATAC_peaks.tsv.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE205nnn/GSE205117/"
        "suppl/GSE205117_ATAC_peaks.tsv.gz"
    ),
    "GCF_000001635.27_genomic.gtf.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/"
        "GCF_000001635.27_GRCm39/GCF_000001635.27_GRCm39_genomic.gtf.gz"
    ),
    "gencode.v44.annotation.gtf.gz": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/"
        "gencode.v44.annotation.gtf.gz"
    ),
}


def download_file(
    url: str, destination: Path, retries: int = 5, delay_sec: int = 5
) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".partial")
    tmp_path.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp_path.replace(destination)
            return
        except urllib.error.URLError as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(delay_sec)


def sanitize_filename(url: str) -> str:
    encoded = url.rsplit("/", 1)[-1]
    decoded = urllib.parse.unquote(encoded)
    return decoded.replace("%5F", "_")


def copy_if_present(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            return True
        shutil.copytree(source, destination)
        return True
    if destination.exists():
        return True
    shutil.copy2(source, destination)
    return True


def ensure_file(
    *,
    url: str,
    destination: Path,
    cache_candidate: Path | None = None,
    prefer_local_cache: bool = True,
) -> str:
    if destination.exists():
        return "exists"
    if (
        prefer_local_cache
        and cache_candidate
        and copy_if_present(cache_candidate, destination)
    ):
        return "copied"
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, destination)
    return "downloaded"


def ensure_gunzip(
    *,
    source_gz: Path,
    destination_plain: Path,
) -> str:
    if destination_plain.exists():
        return "exists"
    destination_plain.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source_gz, "rb") as src, destination_plain.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return "decompressed"


def stage_embryonic(
    output_base: Path,
    cache_base: Path,
    prefer_local_cache: bool,
    samples: set[str] | None = None,
) -> None:
    output_root = output_base / "embryonic" / "raw"
    cache_root = cache_base / "embryonic" / "raw"
    output_root.mkdir(parents=True, exist_ok=True)

    for sample_label, urls in sorted(EMBRYONIC_SAMPLES.items()):
        if samples and sample_label not in samples:
            continue
        sample_dir = output_root / sample_label
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"[embryonic] {sample_label}")
        for url in urls:
            filename = sanitize_filename(url)
            destination = sample_dir / filename
            cache_candidate = cache_root / sample_label / filename
            status = ensure_file(
                url=url,
                destination=destination,
                cache_candidate=cache_candidate,
                prefer_local_cache=prefer_local_cache,
            )
            print(f"  - {status:10s} {destination.name}")

    refs_dir = output_base / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    print("[embryonic] references")
    for filename, url in REFERENCE_URLS.items():
        destination = refs_dir / filename
        status = ensure_file(
            url=url,
            destination=destination,
            cache_candidate=(cache_base / "references" / filename),
            prefer_local_cache=prefer_local_cache,
        )
        print(f"  - {status:10s} {destination.name}")

    mouse_gtf_gz = refs_dir / "GCF_000001635.27_genomic.gtf.gz"
    mouse_gtf_plain = refs_dir / "GCF_000001635.27_genomic.gtf"
    status = ensure_gunzip(source_gz=mouse_gtf_gz, destination_plain=mouse_gtf_plain)
    print(f"  - {status:10s} {mouse_gtf_plain.name}")


def stage_endothelial(
    output_base: Path,
    cache_base: Path,
    prefer_local_cache: bool,
    include_series_archive: bool,
    include_derived_cache: bool,
    samples: set[str] | None = None,
) -> None:
    output_root = output_base / "endothelial" / "raw"
    cache_root = cache_base / "endothelial" / "raw"
    output_root.mkdir(parents=True, exist_ok=True)

    for sample_label, urls in sorted(ENDOTHELIAL_SAMPLES.items()):
        if samples and sample_label not in samples:
            continue
        sample_dir = output_root / sample_label
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"[endothelial] {sample_label}")
        for url in urls:
            filename = sanitize_filename(url)
            destination = sample_dir / filename
            cache_candidate = cache_root / sample_label / filename
            status = ensure_file(
                url=url,
                destination=destination,
                cache_candidate=cache_candidate,
                prefer_local_cache=prefer_local_cache,
            )
            print(f"  - {status:10s} {destination.name}")

    print("[endothelial] supplementary files")
    for filename, url in ENDOTHELIAL_SUPPLEMENTARY_URLS.items():
        destination = output_root / filename
        cache_candidate = cache_root / filename
        status = ensure_file(
            url=url,
            destination=destination,
            cache_candidate=cache_candidate,
            prefer_local_cache=prefer_local_cache,
        )
        print(f"  - {status:10s} {destination.name}")

    if include_series_archive:
        archive_name = sanitize_filename(ENDOTHELIAL_SERIES_ARCHIVE_URL)
        destination = output_root / archive_name
        status = ensure_file(
            url=ENDOTHELIAL_SERIES_ARCHIVE_URL,
            destination=destination,
            cache_candidate=cache_root / archive_name,
            prefer_local_cache=prefer_local_cache,
        )
        print(f"  - {status:10s} {destination.name}")

    if include_derived_cache:
        print("[endothelial] derived cache files")
        for filename in ENDOTHELIAL_DERIVED_CACHE_FILES:
            source = cache_root / filename
            destination = output_root / filename
            if copy_if_present(source, destination):
                print(f"  - {'copied':10s} {destination.name}")
        for dirname in ENDOTHELIAL_DERIVED_CACHE_DIRS:
            source = cache_root / dirname
            destination = output_root / dirname
            if copy_if_present(source, destination):
                print(f"  - {'copied':10s} {destination.name}/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download or stage GEO raw inputs into data/"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("embryonic", "endothelial", "all"),
        default=["all"],
        help="Which datasets to fetch",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        help="Optional sample names to fetch within the selected dataset(s).",
    )
    parser.add_argument(
        "--output-base",
        default=str(Path.cwd() / "data"),
        help="Base destination directory (default: ./data)",
    )
    parser.add_argument(
        "--cache-base",
        default=str(Path.cwd() / "data"),
        help="Existing local cache to reuse before downloading (default: ./data)",
    )
    parser.add_argument(
        "--use-local-cache",
        action="store_true",
        help="Reuse existing files under --cache-base before downloading",
    )
    parser.add_argument(
        "--include-series-archive",
        action="store_true",
        help="Download the large GSE270141_RAW.tar archive for endothelial",
    )
    parser.add_argument(
        "--include-derived-cache",
        action="store_true",
        help="Also copy local derived endothelial cache files (RDS/H5AD/fragments) when available",
    )
    args = parser.parse_args(argv)

    output_base = Path(args.output_base).expanduser().resolve()
    cache_base = Path(args.cache_base).expanduser().resolve()
    prefer_local_cache = args.use_local_cache
    requested = set(args.datasets)
    requested_samples = set(args.samples or [])
    if "all" in requested:
        requested = {"embryonic", "endothelial"}

    if "embryonic" in requested:
        stage_embryonic(
            output_base,
            cache_base,
            prefer_local_cache,
            samples=requested_samples or None,
        )
    if "endothelial" in requested:
        stage_endothelial(
            output_base,
            cache_base,
            prefer_local_cache,
            include_series_archive=args.include_series_archive,
            include_derived_cache=args.include_derived_cache,
            samples=requested_samples or None,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

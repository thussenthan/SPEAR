#!/usr/bin/env python3
"""Pre-flight validation script for SPEAR pipeline runs.

SPEAR: Single-cell-based Prediction of Gene Expression from Chromatin Accessibility Readouts.
Validates environment, data files, dependencies, and configuration before launching
expensive HPC jobs. Run this to catch configuration errors early (SLURM job scripts are optional).

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --base-dir /path/to/project
    python scripts/preflight_check.py --gene-manifest data/embryonic/manifests/selected_genes_100.csv
    python scripts/preflight_check.py --atac-path data/embryonic/processed/combined_ATAC_qc.h5ad
"""

import argparse
import gzip
import importlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(msg: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")


def print_ok(msg: str) -> None:
    print(f"  {Colors.GREEN}✓{Colors.END} {msg}")


def print_warn(msg: str) -> None:
    print(f"  {Colors.YELLOW}⚠{Colors.END} {msg}")


def print_error(msg: str) -> None:
    print(f"  {Colors.RED}✗{Colors.END} {msg}")


def check_python_version() -> bool:
    """Verify Python version >= 3.10."""
    version = sys.version_info
    if version >= (3, 10):
        print_ok(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    print_error(f"Python {version.major}.{version.minor} < 3.10 (required)")
    return False


def check_required_packages() -> Tuple[bool, List[str]]:
    """Verify all required packages are installed."""
    required = [
        "anndata",
        "catboost",
        "numpy",
        "pandas",
        "psutil",
        "scipy",
        "sklearn",
        "shap",
        "torch",
        "xgboost",
        "matplotlib",
        "seaborn",
        "joblib",
    ]

    missing = []
    for pkg in required:
        try:
            if pkg == "sklearn":
                importlib.import_module("sklearn")
            else:
                importlib.import_module(pkg)
            print_ok(f"Package '{pkg}' installed")
        except ImportError:
            print_error(f"Package '{pkg}' missing")
            missing.append(pkg)

    # scanpy is optional: SPEAR no longer depends on it for manifest generation.
    try:
        importlib.import_module("scanpy")
        print_ok("Package 'scanpy' installed (optional)")
    except Exception as exc:
        print_warn(f"Package 'scanpy' unavailable (optional): {exc}")

    return len(missing) == 0, missing


def check_optional_packages() -> None:
    """Check optional packages and warn if missing.

    This function is intentionally kept as a stub for future optional
    dependencies. Add entries to the ``optional`` dict in the form
    ``{"package_name": "why it's useful"}`` when such dependencies are
    introduced.
    """

    optional = {}

    for pkg, warning in optional.items():
        try:
            importlib.import_module(pkg)
            print_ok(f"Optional package '{pkg}' installed")
        except ImportError:
            print_warn(f"Optional package '{pkg}' missing ({warning})")


def check_file_exists(path: Path, description: str) -> bool:
    """Check if file exists and is readable."""
    if not path.exists():
        print_error(f"{description} not found: {path}")
        return False
    if not path.is_file():
        print_error(f"{description} is not a file: {path}")
        return False
    try:
        # Test readability
        with path.open("rb") as f:
            f.read(1)
        print_ok(f"{description}: {path}")
        return True
    except Exception as exc:
        print_error(f"{description} not readable: {path} ({exc})")
        return False


def check_directory_writable(path: Path, description: str) -> bool:
    """Check if directory exists and is writable."""
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            print_ok(f"{description} created: {path}")
            return True
        except Exception as exc:
            print_error(f"Cannot create {description}: {path} ({exc})")
            return False

    if not path.is_dir():
        print_error(f"{description} exists but is not a directory: {path}")
        return False

    # Test write permission
    test_file = path / ".preflight_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print_ok(f"{description} writable: {path}")
        return True
    except Exception as exc:
        print_error(f"{description} not writable: {path} ({exc})")
        return False


def check_h5ad_file(path: Path, description: str) -> bool:
    """Validate AnnData h5ad file."""
    if not check_file_exists(path, description):
        return False

    try:
        import anndata as ad

        adata = ad.read_h5ad(path.as_posix())
        n_obs, n_vars = adata.shape
        print_ok(f"  {description} loaded: {n_obs:,} cells × {n_vars:,} features")

        # Check for common issues
        if n_obs == 0:
            print_warn(f"  {description} has 0 observations")
        if n_vars == 0:
            print_warn(f"  {description} has 0 variables")

        return True
    except Exception as exc:
        print_error(f"  Failed to load {description}: {exc}")
        return False


def check_gtf_file(path: Path) -> bool:
    """Validate GTF file structure."""
    if not check_file_exists(path, "GTF annotation file"):
        return False

    try:
        opener = gzip.open if path.suffix == ".gz" else open
        gene_count = 0
        with opener(path, "rt") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                if parts[2] == "gene":
                    gene_count += 1
                if gene_count >= 10:  # Sample first 10 genes
                    break

        if gene_count > 0:
            print_ok(f"  GTF file parseable (sampled {gene_count} genes)")
            return True
        else:
            print_warn("  GTF file contains no gene entries")
            return False
    except Exception as exc:
        print_error(f"  Failed to parse GTF: {exc}")
        return False


def check_gene_manifest(path: Optional[Path]) -> bool:
    """Validate gene manifest file."""
    if path is None:
        print_warn("No gene manifest specified (will use all genes)")
        return True

    if not check_file_exists(path, "Gene manifest"):
        return False

    try:
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        # Skip header if present
        if lines and any(
            keyword in lines[0].lower() for keyword in ["gene", "symbol", "name"]
        ):
            lines = lines[1:]

        if len(lines) == 0:
            print_error("  Gene manifest is empty")
            return False

        print_ok(f"  Gene manifest contains {len(lines)} genes")
        if len(lines) > 0:
            print_ok(f"  First gene: {lines[0]}")
        return True
    except Exception as exc:
        print_error(f"  Failed to read gene manifest: {exc}")
        return False


def check_pipeline_package() -> bool:
    """Verify spear package is importable."""
    try:
        import spear  # noqa: F401
        from spear import config, data, evaluation, models, training  # noqa: F401

        print_ok("Pipeline package 'spear' importable")
        print_ok("  Core modules: config, data, models, training, evaluation")
        return True
    except ImportError as exc:
        print_error(f"Pipeline package import failed: {exc}")
        print_error("  Run: pip install -e . from project root")
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight validation for SPEAR pipeline runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        default=Path.cwd(),
        type=Path,
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--atac-path",
        type=Path,
        help="Override ATAC AnnData path (default: auto-detect in base-dir)",
    )
    parser.add_argument(
        "--rna-path",
        type=Path,
        help="Override RNA AnnData path (default: auto-detect in base-dir)",
    )
    parser.add_argument(
        "--gtf-path",
        type=Path,
        help="Override GTF path (default: data/references/GCF_000001635.27_genomic.gtf)",
    )
    parser.add_argument(
        "--gene-manifest",
        type=Path,
        help="Gene manifest file to validate",
    )
    parser.add_argument(
        "--skip-data-checks",
        action="store_true",
        help="Skip AnnData file validation (faster)",
    )

    args = parser.parse_args(argv)
    base_dir = args.base_dir.expanduser().resolve()

    print_header("SPEAR Pipeline Pre-flight Check")
    print(f"Base directory: {base_dir}\n")

    results = []

    # 1. Environment checks
    print_header("1. Environment Validation")
    results.append(check_python_version())
    pkg_ok, missing = check_required_packages()
    results.append(pkg_ok)

    check_optional_packages()
    results.append(check_pipeline_package())

    # 2. Directory structure
    print_header("2. Directory Structure")
    output_dir = base_dir / "output"
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"

    results.append(check_directory_writable(results_dir, "Results directory"))
    results.append(check_directory_writable(logs_dir, "Logs directory"))

    # 3. Data files
    print_header("3. Data Files")

    # Auto-detect data paths
    if args.atac_path:
        atac_path = args.atac_path.expanduser().resolve()
    else:
        # Try common locations
        candidates = [
            base_dir / "data" / "embryonic" / "processed" / "combined_ATAC_qc.h5ad",
            base_dir
            / "data"
            / "endothelial"
            / "processed"
            / "combined_ATAC_qc_under15%mito.h5ad",
            base_dir / "combined_ATAC_qc.h5ad",
        ]
        atac_path = next((p for p in candidates if p.exists()), candidates[0])

    if args.rna_path:
        rna_path = args.rna_path.expanduser().resolve()
    else:
        candidates = [
            base_dir / "data" / "embryonic" / "processed" / "combined_RNA_qc.h5ad",
            base_dir
            / "data"
            / "endothelial"
            / "processed"
            / "combined_RNA_qc_under15%mito.h5ad",
            base_dir / "combined_RNA_qc.h5ad",
        ]
        rna_path = next((p for p in candidates if p.exists()), candidates[0])

    if args.gtf_path:
        gtf_path = args.gtf_path.expanduser().resolve()
    else:
        gtf_path = base_dir / "data" / "references" / "GCF_000001635.27_genomic.gtf"

    if not args.skip_data_checks:
        results.append(check_h5ad_file(atac_path, "ATAC AnnData"))
        results.append(check_h5ad_file(rna_path, "RNA AnnData"))
    else:
        results.append(check_file_exists(atac_path, "ATAC AnnData"))
        results.append(check_file_exists(rna_path, "RNA AnnData"))

    results.append(check_gtf_file(gtf_path))
    results.append(check_gene_manifest(args.gene_manifest))

    # 4. SLURM scripts (optional)
    print_header("4. SLURM Job Scripts (Optional)")
    cpu_script = base_dir / "jobs" / "slurm_spear_cellwise_chunked.sbatch"
    gpu_script = base_dir / "jobs" / "slurm_spear_cellwise_chunked_gpu.sbatch"
    if cpu_script.exists():
        results.append(check_file_exists(cpu_script, "CPU SLURM script"))
    else:
        print_warn(f"CPU SLURM script not found (optional): {cpu_script}")
    if gpu_script.exists():
        results.append(check_file_exists(gpu_script, "GPU SLURM script"))
    else:
        print_warn(f"GPU SLURM script not found (optional): {gpu_script}")

    # Summary
    print_header("Summary")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}✓ All checks passed ({passed}/{total}){Colors.END}"
        )
        print(f"{Colors.GREEN}Pipeline is ready for production runs.{Colors.END}\n")
        return 0
    else:
        failed = total - passed
        print(
            f"\n{Colors.YELLOW}{Colors.BOLD}⚠ {failed} check(s) failed ({passed}/{total} passed){Colors.END}"
        )
        print(
            f"{Colors.YELLOW}Please address the issues above before launching HPC jobs.{Colors.END}\n"
        )

        if not pkg_ok and missing:
            print(f"{Colors.YELLOW}Install missing packages:{Colors.END}")
            print(f"  pip install {' '.join(missing)}\n")

        return 1


if __name__ == "__main__":
    sys.exit(main())

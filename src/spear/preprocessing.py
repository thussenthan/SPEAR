from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

import anndata as ad
import pandas as pd


def align_modalities(
    rna: ad.AnnData, atac: ad.AnnData
) -> tuple[ad.AnnData, ad.AnnData]:
    """Subset paired modalities to a shared, ordered cell index."""

    common = pd.Index(rna.obs_names).intersection(pd.Index(atac.obs_names))
    if common.empty:
        raise ValueError("RNA and ATAC modalities do not share any cell barcodes")
    return rna[common].copy(), atac[common].copy()


def build_mudata(rna: ad.AnnData, atac: ad.AnnData):
    """Create a MuData object lazily so muon is only required when used."""

    mu = import_real_muon()

    rna, atac = align_modalities(rna, atac)
    return mu.MuData({"rna": rna, "atac": atac})


def import_real_muon():
    """Import the installed muon package instead of the local tutorial directory."""

    project_root = Path(__file__).resolve().parents[2]
    original_sys_path = list(sys.path)
    try:
        sys.path = [
            entry for entry in sys.path if Path(entry or ".").resolve() != project_root
        ]
        mu = importlib.import_module("muon")
    finally:
        sys.path = original_sys_path

    if not hasattr(mu, "MuData"):
        raise ImportError(
            "Resolved 'muon' does not expose MuData. "
            "Install the muon package and avoid importing the local tutorial directory."
        )
    return mu


def export_spear_ready(
    rna: ad.AnnData,
    atac: ad.AnnData,
    output_dir: Path,
    prefix: str = "combined",
    split_fields: Iterable[str] | None = None,
    write_h5mu: bool = False,
) -> dict[str, Path]:
    """Write combined paired AnnData outputs plus optional split subsets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rna, atac = align_modalities(rna, atac)

    outputs = {
        "rna": output_dir / f"{prefix}_RNA_muon_qc.h5ad",
        "atac": output_dir / f"{prefix}_ATAC_muon_qc.h5ad",
    }
    rna.write_h5ad(outputs["rna"])
    atac.write_h5ad(outputs["atac"])

    if write_h5mu:
        h5mu_path = output_dir / f"{prefix}_muon_qc.h5mu"
        build_mudata(rna, atac).write_h5mu(h5mu_path)
        outputs["h5mu"] = h5mu_path

    for field in split_fields or ():
        if field not in rna.obs:
            continue
        values = pd.Index(rna.obs[field].dropna().astype(str).unique()).sort_values()
        for value in values:
            mask = rna.obs[field].astype(str).eq(value)
            if not mask.any():
                continue
            safe_value = value.replace("/", "_").replace(" ", "_")
            subset_dir = output_dir / "splits" / field / safe_value
            subset_dir.mkdir(parents=True, exist_ok=True)
            subset_rna = rna[mask].copy()
            subset_atac = atac[mask].copy()
            subset_rna.write_h5ad(subset_dir / f"{prefix}_RNA_muon_qc.h5ad")
            subset_atac.write_h5ad(subset_dir / f"{prefix}_ATAC_muon_qc.h5ad")

    return outputs

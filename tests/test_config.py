from pathlib import Path

import pytest

from spear.config import PipelineConfig, PathsConfig


def _pipeline_config(cache_scope: str) -> PipelineConfig:
    root = Path("/tmp/spear-test")
    return PipelineConfig(
        paths=PathsConfig(
            base_dir=root,
            atac_path=root / "atac.h5ad",
            rna_path=root / "rna.h5ad",
            gtf_path=root / "genes.gtf",
            output_dir=root / "output",
            logs_dir=root / "logs",
            figures_dir=root / "figs",
        ),
        cache_dir=root / ".spear_cache",
        cache_scope=cache_scope,
    )


def test_cache_scope_auto_uses_only_cellwise_disk_cache() -> None:
    config = _pipeline_config("auto")

    assert config.cache_dir_for_scope("cellwise") == Path(
        "/tmp/spear-test/.spear_cache"
    )
    assert config.cache_dir_for_scope("gene") is None


def test_cache_scope_gene_opts_into_per_gene_disk_cache() -> None:
    config = _pipeline_config("gene")

    assert config.cache_dir_for_scope("gene") == Path("/tmp/spear-test/.spear_cache")
    assert config.cache_dir_for_scope("cellwise") is None


def test_cache_scope_rejects_unknown_value() -> None:
    config = _pipeline_config("invalid")

    with pytest.raises(ValueError, match="cache_scope"):
        config.cache_dir_for_scope("gene")


def test_all_models_normalizes_scalar_model_names() -> None:
    config = _pipeline_config("auto")
    config.models.model_names = "ridge"  # type: ignore[assignment]

    assert config.all_models() == ["ridge"]

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_preflight_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / ("preflight_wandb_sweeps.py")
    )
    spec = importlib.util.spec_from_file_location("preflight_wandb_sweeps", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["gene_name"]
    lines.extend(f"Gene{i}" for i in range(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_minimal_sweep(
    path: Path,
    *,
    manifest: str = "data/manifests/test_random_2.csv",
    model: str = "transformer",
) -> None:
    path.write_text(
        f"""
method: grid
metric:
  goal: maximize
  name: test_pearson
parameters:
  feature_basis:
    values: ["bin", "peak"]
  global_atac_components:
    values: [0, 10]
command:
  - ${{env}}
  - python
  - -m
  - spear.cli
  - --base-dir
  - .
  - --models
  - {model}
  - --wandb
  - --wandb-sweep
  - --wandb-group
  - Q1-feature-representation
  - --wandb-tags
  - Q1-feature-representation
  - per-gene
  - --atac-path
  - data/test_ATAC.h5ad
  - --rna-path
  - data/test_RNA.h5ad
  - --gene-manifest
  - {manifest}
  - --cache-dir
  - data/.spear_cache
""".lstrip(),
        encoding="utf-8",
    )


def test_grid_run_count_and_run_cap() -> None:
    module = _load_preflight_module()

    grid_sweep = {
        "method": "grid",
        "parameters": {
            "a": {"values": [1, 2]},
            "b": {"values": ["x", "y", "z"]},
            "fixed": {"value": 1},
        },
    }
    capped_sweep = {"method": "random", "run_cap": 36, "parameters": {}}

    assert module.expected_run_count("QX", grid_sweep, {"QX": 6}) == 6
    assert module.expected_run_count("Q6", capped_sweep) == 36


def test_valid_sweep_paths_and_counts(tmp_path: Path) -> None:
    module = _load_preflight_module()

    (tmp_path / "data").mkdir()
    (tmp_path / "data/test_ATAC.h5ad").write_text("fake", encoding="utf-8")
    (tmp_path / "data/test_RNA.h5ad").write_text("fake", encoding="utf-8")
    (tmp_path / "data/.spear_cache").mkdir()
    _write_manifest(tmp_path / "data/manifests/test_random_2.csv", rows=2)
    sweep_path = tmp_path / "Q1_feature_representation.yaml"
    _write_minimal_sweep(sweep_path)

    summary, issues = module.validate_sweep_file(
        sweep_path, base_dir=tmp_path, expected_counts={"Q1": 4}
    )

    assert summary is not None
    assert summary.expected_runs == 4
    assert summary.manifest_gene_count == 2
    assert issues == []


def test_manifest_count_mismatch_is_error(tmp_path: Path) -> None:
    module = _load_preflight_module()

    (tmp_path / "data").mkdir()
    (tmp_path / "data/test_ATAC.h5ad").write_text("fake", encoding="utf-8")
    (tmp_path / "data/test_RNA.h5ad").write_text("fake", encoding="utf-8")
    (tmp_path / "data/.spear_cache").mkdir()
    _write_manifest(tmp_path / "data/manifests/test_random_1000.csv", rows=2)
    sweep_path = tmp_path / "Q1_feature_representation.yaml"
    _write_minimal_sweep(sweep_path, manifest="data/manifests/test_random_1000.csv")

    _, issues = module.validate_sweep_file(
        sweep_path, base_dir=tmp_path, expected_counts={"Q1": 4}
    )

    assert any("Manifest row count mismatch" in issue.message for issue in issues)


def test_unknown_model_is_error(tmp_path: Path) -> None:
    module = _load_preflight_module()

    (tmp_path / "data").mkdir()
    (tmp_path / "data/test_ATAC.h5ad").write_text("fake", encoding="utf-8")
    (tmp_path / "data/test_RNA.h5ad").write_text("fake", encoding="utf-8")
    (tmp_path / "data/.spear_cache").mkdir()
    _write_manifest(tmp_path / "data/manifests/test_random_2.csv", rows=2)
    sweep_path = tmp_path / "Q1_feature_representation.yaml"
    _write_minimal_sweep(sweep_path, model="not_a_model")

    _, issues = module.validate_sweep_file(
        sweep_path, base_dir=tmp_path, expected_counts={"Q1": 4}
    )

    assert any("Unknown model names" in issue.message for issue in issues)


def test_model_substitution_is_rejected_for_wandb_sweeps(tmp_path: Path) -> None:
    module = _load_preflight_module()

    (tmp_path / "data").mkdir()
    (tmp_path / "data/test_ATAC.h5ad").write_text("fake", encoding="utf-8")
    (tmp_path / "data/test_RNA.h5ad").write_text("fake", encoding="utf-8")
    (tmp_path / "data/.spear_cache").mkdir()
    _write_manifest(tmp_path / "data/manifests/test_random_2.csv", rows=2)
    sweep_path = tmp_path / "Q1_feature_representation.yaml"
    _write_minimal_sweep(sweep_path, model="${model}")

    _, issues = module.validate_sweep_file(
        sweep_path, base_dir=tmp_path, expected_counts={"Q1": 4}
    )

    assert any("must not use ${model}" in issue.message for issue in issues)


def test_multi_model_sweep_value_is_rejected(tmp_path: Path) -> None:
    module = _load_preflight_module()

    (tmp_path / "data").mkdir()
    (tmp_path / "data/test_ATAC.h5ad").write_text("fake", encoding="utf-8")
    (tmp_path / "data/test_RNA.h5ad").write_text("fake", encoding="utf-8")
    (tmp_path / "data/.spear_cache").mkdir()
    _write_manifest(tmp_path / "data/manifests/test_random_2.csv", rows=2)
    sweep_path = tmp_path / "Q1_feature_representation.yaml"
    _write_minimal_sweep(sweep_path)
    text = sweep_path.read_text(encoding="utf-8")
    text = text.replace(
        'feature_basis:\n    values: ["bin", "peak"]',
        'model_names:\n    values:\n      - ["ridge", "mlp"]',
    )
    sweep_path.write_text(text, encoding="utf-8")

    _, issues = module.validate_sweep_file(
        sweep_path, base_dir=tmp_path, expected_counts={"Q1": 2}
    )

    assert any("exactly one model per run" in issue.message for issue in issues)

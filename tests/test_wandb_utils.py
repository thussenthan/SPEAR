from __future__ import annotations

from pathlib import Path

from spear.config import PipelineConfig, PathsConfig
from spear.wandb_utils import (
    apply_sweep_overrides,
    model_used_fallback,
    run_has_fallbacks,
    wandb_finish,
)


class _FakeRun:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.summary = {}
        self.exit_code = None

    def finish(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code


def _pipeline_config() -> PipelineConfig:
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
        )
    )


def test_model_used_fallback_detects_explicit_flag() -> None:
    assert model_used_fallback({"used_fallback": True}) is True


def test_model_used_fallback_detects_reasons() -> None:
    assert model_used_fallback({"fallback_reasons": ["dummy fallback"]}) is True


def test_run_has_fallbacks_detects_any_model() -> None:
    model_run_details = {
        "cnn": {"status": "succeeded"},
        "rnn": {"status": "failed", "fallbacks": ["gene: reason"]},
    }
    assert run_has_fallbacks(model_run_details) is True


def test_wandb_finish_marks_failed_runs_with_nonzero_exit_code() -> None:
    run = _FakeRun()
    wandb_finish(run, status="failed")
    assert run.exit_code == 1


def test_apply_sweep_overrides_normalizes_scalar_model_names() -> None:
    config = _pipeline_config()
    config.wandb.sweep_overrides = True
    run = _FakeRun({"model_names": "ridge"})

    assert apply_sweep_overrides(config, run) is True
    assert config.models.model_names == ["ridge"]
    assert config.all_models() == ["ridge"]

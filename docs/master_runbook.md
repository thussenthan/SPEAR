# SPEAR End-to-End Runbook (Single-cell-based Prediction of Gene Expression from Chromatin Accessibility Readouts)

## Overview

This runbook connects the biological motivation for the mouse embryonic stem cell (mESC) project with the computational workflow implemented in the SPEAR repository. Follow the stages in order when onboarding a new environment, reproducing results, or preparing a public release.

## Inputs

- SPEAR repository checkout.
- Access to GEO data referenced in dataset docs.
- Python 3.10+ environment with project dependencies.

## Outputs

- Model checkpoints, metrics, summaries, and figures under `output/` and `analysis/figs`.
- Log files under `output/logs` (pipeline logs: `<run_name>.log`; scheduler logs commonly appear as `spear_<jobid>_<task>.out/.err` when using job arrays).
- Default run naming: `spear_<model>_<genes>_<dataset>_<cpu/gpu>_<timestamp>`; W&B run names default to `<model>_<genes>_<dataset>` unless overridden.

## Usage

### Orientation

- Scope: infer gene regulatory programs from paired single-cell RNA-seq and ATAC-seq, using the included mouse embryonic and endothelial examples as reference workflows.
- Target users: computational biologists and ML practitioners working with paired single-cell ATAC/RNA data on local workstations, cloud instances, or HPC environments.
- Output: model checkpoints, per-gene metrics, aggregate summaries, and narrative analyses ready for publication.

### Biological frame

- mESC differentiation offers a controlled setting to study lineage priming and regulatory switches.
- Multi-omic pairing ensures perturbations in chromatin accessibility can be linked to transcriptional output.
- Replicate structure (two per stage when available) supports variance estimation and biological replicability.

### Computational frame

- Data volume (10x matrices and ATAC fragment files) requires staged downloads and curated AnnData objects.
- Training splits gene targets into HPC-friendly chunks and schedules heterogeneous model families.
- Results consolidation and metadata capture are necessary for reproducibility and comparative modeling.

### Key data products

- Raw GEO downloads and curated AnnData layouts are documented in `docs/mouse_esc_dataset.md` and `docs/endothelial_dataset.md`.
- Gene manifests live under `data/embryonic/manifests/` and define target scopes for each run.
- Reference annotations are kept in `data/references/`.

### Stage 1 - Bootstrap Environment

### Stage 1 - Practical steps

- Clone the repository into `$HOME` or project scratch.
- Install Conda or Mamba if not already available.
- Create and activate a clean Python 3.10+ environment (Conda/Mamba optional).
- Install dependencies and the project in editable mode: `pip install -r requirements.txt` then `pip install -e .`.

### Stage 1 - Computer science perspective

- Pinning to the provided requirements file locks ML library versions needed for GPU training and consistent serialization; match Python/CUDA to your hardware.
- Editable installs ensure CLI entry points and module imports resolve to the current working tree, simplifying iterative development.

### Stage 1 - Biology perspective

- Stable environments reduce the risk of numerical drift in downstream metrics, allowing direct biological comparisons with published results.
- Maintaining the same dependency stack as the original analysis preserves behavior of preprocessing routines that enforce biologically motivated QC filters.

### Stage 2 - Acquire and Stage Data

### Stage 2 - Practical steps

- Review `docs/mouse_esc_dataset.md` or `docs/endothelial_dataset.md` for provenance and sample inventory.
- Download raw data from GEO accession GSE205117 and place in `data/embryonic/raw/`.
- Preprocess data using scripts in `scripts/` to generate AnnData files in `data/embryonic/processed/`.
- Ensure reference GTF files are present in `data/references/`.

### Stage 2 - Computer science perspective

- The download script is idempotent: files already present are skipped, preventing accidental re-transfer of large archives.
- Consistent directory layout enables automation in Snakemake-style pipelines and simple globbing in analysis scripts.
- Validating AnnData objects guards against schema mismatches that would break downstream ingestion.

### Stage 2 - Biology perspective

- Confirming each replicate directory contains both RNA and ATAC modalities preserves the paired design necessary for integrative inference.
- Refreshing processed matrices after raw updates avoids mixing batches generated with different cell filters or peak calling parameters.

### Stage 3 - Understand Configuration Surface

### Stage 3 - Practical steps

- Review `README.md` for the current pipeline behavior and defaults.
- Consult `docs/config_reference.md` for every CLI flag supported by `src`.
- Determine the gene manifest(s), chromosome scope, window size, and training overrides for your planned run; record these in `todo.md` or a run sheet.
- For ResNet experiments, decide whether to keep squeeze-excitation attention (`--resnet-attention se`, default) or run an ablation with `--resnet-attention none`; tune bottleneck capacity with `--resnet-attention-se-reduction`.
- For SVR runs, note that `TrainingConfig` exposes `svr_kernel`, `svr_C`, `svr_epsilon`, `svr_max_iter`, and `svr_tol` with defaults documented in `docs/config_reference.md`.

### Stage 3 - Computer science perspective

- Many script parameters have sensible defaults but interact (e.g., `--multi-output` with chunk count); reviewing the reference prevents invalid combinations.
- Knowing the configuration space upfront facilitates reproducibility by enabling exact command reconstruction from logged YAML/JSON artifacts.

### Stage 3 - Biology perspective

- Deciding on gene subsets (pan-cellular vs lineage-specific) and genomic windows frames the biological hypotheses each run can test.
- Clarifying perturbation cohorts (wild-type vs CRISPR) ahead of time ensures downstream comparisons remain interpretable.

### Stage 4 - Internalize Pipeline Architecture

Refer to the "Preprocessing Details", "Supported Models", and "Results & Visualization" sections in `README.md` for the pipeline walkthrough. This runbook stays focused on operational steps and release prep.

### Stage 5 - Prime Output and Metadata Directories

### Stage 5 - Practical steps

- Ensure output directories exist (`mkdir -p` as needed).
- Prepare a dedicated log directory if cluster policy requires absolute paths distinct from the repo.
- Confirm that storage quotas can accommodate model checkpoints, particularly for Transformer runs.

### Stage 5 - Computer science perspective

- Pre-creating directories avoids race conditions in batch jobs and keeps scheduler output organized.
- Planning storage usage prevents silent failures caused by quota exhaustion mid-training.

### Stage 5 - Biology perspective

- Organized outputs streamline later aggregation into figures and tables for manuscripts.
- Maintaining log lineage enables provenance tracking when reported biological insights rely on specific training runs.

### Stage 6 - Run Local Smoke Tests

### Stage 6 - Practical steps

- Execute a CPU-only smoke run (via `spear` or module form):  
  `spear --models mlp --gene-manifest data/embryonic/manifests/selected_genes_10.csv --device cpu --k-folds 2 --epochs 2 --run-name dev_smoke_local`
- Confirm outputs are generated successfully.
- Inspect logs for import errors, missing data references, or serialization issues.
- You can also run `python scripts/preflight_check.py` to validate environment, package availability, and data paths (AnnData/GTF) before queueing jobs.

### Stage 6 - Computer science perspective

- Early detection of dependency or path problems saves GPU queue time and ensures serialization schemas match expectations.

### Stage 6 - Biology perspective

- Even miniature runs validate that cell metadata lines up with gene manifests (e.g., no empty matrices), avoiding biological misinterpretations later.

### Stage 7 - Submit Batch Jobs

### Stage 7 - Practical steps

- Use the templates in `jobs/` as starting points and adapt account, partition/queue, resources, and launcher commands for your environment. Map array indices to `(model, chunk)` combinations consistent with your manifest size.

### Stage 7 - Computer science perspective

- Job arrays map deterministically to `(model, chunk)` pairs; log files encoded with array indices make troubleshooting parallel jobs tractable.
- Explicit environment variables keep submission commands self-documenting and reusable in automation scripts.

### Stage 7 - Biology perspective

- Running diverse model classes provides complementary evidence of regulatory influence (e.g., nonlinear vs linear importance patterns).
- Pairing runs with specific manifests (e.g., endothelial vs global genes) focuses the analysis on biologically coherent questions.

### Stage 8 - Monitor Execution and Validate Intermediate Outputs

### Stage 8 - Practical steps

- Track queued/running jobs with your scheduler CLI (for example: `squeue`, `qstat`, or `bjobs`) and tail log files under `output/logs`.
- Spot-check GPU utilization with `nvidia-smi` or scheduler-integrated monitoring commands when available.
- Confirm each chunk writes `metrics_per_gene.csv`, `summary_metrics.csv`, and `run_configuration.json`.
- Retry failed array indices after addressing the root cause.

### Stage 8 - Computer science perspective

- Monitoring ensures convergence issues, memory exhaustion, or library mismatches are caught promptly.
- Captured configuration snapshots serve as ground truth when regenerating results or sharing with collaborators.

### Stage 8 - Biology perspective

- Mid-run checks verify that metrics fall within expected biological ranges (e.g., correlation coefficients not trivially zero), preventing wasted compute on pathological settings.

### Stage 9 - Aggregate, Interpret, and Visualize

### Stage 9 - Practical steps

- Use scripts in `scripts/` (e.g., `combine_chunk_results.py`) or notebooks under `analysis/` to merge chunk outputs.
- Generate model comparison plots and feature importance summaries; store figures in `analysis/figs`.
- Document findings in Markdown or notebooks, referencing run names and configuration hashes.

### Stage 9 - Computer science perspective

- Consistent aggregation pipelines avoid manual copy/paste errors and support reproducible figure regeneration.
- Persisting intermediate tables enables future statistical testing or meta-analysis without re-running heavy jobs.

### Stage 9 - Biology perspective

- Examine whether top regulatory features align with known developmental regulators, CRISPR perturbation expectations, or spatial gradients.
- Compare accessibility-weighted features against gene expression shifts to propose mechanistic hypotheses.

### Stage 10 - Ensure Reproducibility and Prepare for Release

### Stage 10 - Practical steps

- Update documentation (README, dataset notes, this runbook) with any deviations or newly supported workflows.
- Remove or mask institutional identifiers before sharing publicly.
- Package final results with metadata: run name, date, commit hash, environment spec, and data provenance.
- Optionally, register key outputs in persistent storage (lab archive or institutional repository).

### Stage 10 - Computer science perspective

- Capturing commit hashes and environment files enables deterministic reruns, a prerequisite for confident software releases.
- Reviewing scripts for hard-coded paths or user-specific assumptions prevents portability issues.

### Stage 10 - Biology perspective

- Contextual notes describing biological interpretation, quality thresholds, and open questions transform raw metrics into actionable insights for collaborators.
- Proper provenance documentation supports peer review, future integrative analyses, and compliance with data-sharing policies.

## References

- Dataset specifics and download automation: `docs/mouse_esc_dataset.md`, `docs/endothelial_dataset.md`
- Configuration dictionary: `docs/config_reference.md`
- Job submission templates are provided in `jobs/` and should be customized for your infrastructure.

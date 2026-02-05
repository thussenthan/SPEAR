# SPEAR Package (Single-cell-based Prediction of Gene Expression from Chromatin Accessibility Readouts)

## Overview

Python package housing the reusable components that power the SPEAR workflow.

## Inputs

- Installed dependencies from `requirements.txt`.
- Project configuration via CLI or JSON.

## Outputs

- Python package modules used by the CLI and scripts.

## Usage

Key modules:

- `config.py` – dataclasses describing filesystem layout, training hyperparameters, and model selections.
- `cli.py` – entrypoint for command-line execution (`spear` or `python -m spear.cli`).
- `data.py`, `training.py`, `evaluation.py`, `metrics.py` – data handling, model training loops, and evaluation logic.
- `visualization.py` – plotting utilities for diagnostic figures.

Install in editable mode for development:

```bash
pip install -e .
```

Run `spear --help` (or `python -m spear.cli --help`) for the full list of pipeline options.

## References

- `README.md`

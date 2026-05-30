#!/usr/bin/env python3
"""Emit the canonical sample order for data preprocessing arrays."""

from __future__ import annotations

import argparse

EMBRYONIC_SAMPLES = [
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

ENDOTHELIAL_SAMPLES = [
    "S3H_hypoxia",
    "S3N_normoxia",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("embryonic", "endothelial", "all"),
        default=["all"],
        help="Optional dataset filter.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print only the total number of selected samples.",
    )
    return parser.parse_args()


def iter_rows(datasets: set[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if "embryonic" in datasets:
        rows.extend(("embryonic", sample) for sample in EMBRYONIC_SAMPLES)
    if "endothelial" in datasets:
        rows.extend(("endothelial", sample) for sample in ENDOTHELIAL_SAMPLES)
    return rows


def main() -> int:
    args = parse_args()
    requested = set(args.datasets)
    if "all" in requested:
        requested = {"embryonic", "endothelial"}
    rows = iter_rows(requested)
    if args.count:
        print(len(rows))
        return 0
    for dataset, sample in rows:
        print(f"{dataset}\t{sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

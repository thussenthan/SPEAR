#!/usr/bin/env python3
"""
Combine per-chunk training outputs into unified result folders.

Typical usage:

    python scripts/combine_chunk_results.py \
        --input-root output/results \
        --run-prefix spear_1000genes_k5_pg20_20251106 \
        --include-predictions

The script scans `input-root` for directories produced by chunked Slurm jobs
(`RUN_NAME_chunkXofY_MODEL`). For each `(RUN_NAME, MODEL)` pair it concatenates
`metrics_per_gene.csv`, recomputes aggregate metrics, merges `selected_genes.csv`,
and optionally concatenates `predictions_raw.csv`. Combined artifacts are written
to `output-root/RUN_NAME_MODEL` mirroring non-chunked runs.
"""


import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

RUN_NAME_PATTERN = re.compile(
    r"^(?P<base>.+)_chunk(?P<chunk_index>\d+)of(?P<chunk_total>\d+)_(?P<model>.+)$"
)


@dataclass(frozen=True)
class ChunkRecord:
    base_name: str
    model: str
    chunk_index: int
    chunk_total: int
    run_dir: Path

    @property
    def model_dir(self) -> Path:
        return self.run_dir / "models" / self.model

    @property
    def run_name(self) -> str:
        return self.run_dir.name


def parse_chunk_directory(path: Path) -> Optional[ChunkRecord]:
    match = RUN_NAME_PATTERN.match(path.name)
    if not match:
        return None
    base = match.group("base")
    chunk_index = int(match.group("chunk_index"))
    chunk_total = int(match.group("chunk_total"))
    model = match.group("model")
    return ChunkRecord(
        base_name=base,
        model=model,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        run_dir=path,
    )


def discover_chunks(
    root: Path,
    run_prefixes: Optional[set[str]] = None,
    models: Optional[set[str]] = None,
) -> dict[tuple[str, str], list[ChunkRecord]]:
    grouped: dict[tuple[str, str], list[ChunkRecord]] = defaultdict(list)
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        record = parse_chunk_directory(entry)
        if record is None:
            continue
        if run_prefixes and record.base_name not in run_prefixes:
            continue
        if models and record.model not in models:
            continue
        grouped[(record.base_name, record.model)].append(record)
    return grouped


def load_dataframe(path: Path, add_cols: Optional[dict[str, int | str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if add_cols:
        for key, value in add_cols.items():
            df[key] = value
    return df


def concat_and_dedupe(
    frames: Iterable[pd.DataFrame],
    subset: Optional[list[str]] = None,
) -> pd.DataFrame:
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if subset:
        duplicated = combined.duplicated(subset=subset, keep=False)
        if duplicated.any():
            grouped = (
                combined.groupby(subset, dropna=False)
                .mean(numeric_only=True)
                .reset_index()
            )
            remaining_cols = [col for col in combined.columns if col not in subset]
            for col in remaining_cols:
                if col in grouped.columns:
                    continue
                grouped[col] = combined.groupby(subset, dropna=False)[col].first().values
            return grouped
        combined = combined.drop_duplicates(subset=subset, keep="first")
    return combined.reset_index(drop=True)


def compute_split_means(metrics_per_gene: pd.DataFrame) -> pd.DataFrame:
    if metrics_per_gene.empty:
        return pd.DataFrame(
            columns=["split", "mse", "rmse", "mae", "r2", "spearman", "pearson"]
        )
    numeric_cols = [
        col
        for col in metrics_per_gene.columns
        if col
        not in {
            "gene",
            "split",
            "chunk_index",
        }
    ]
    grouped = (
        metrics_per_gene.groupby("split", dropna=False)[numeric_cols]
        .mean()
        .reset_index()
    )
    return grouped.sort_values("split").reset_index(drop=True)


def combine_cv_metrics(cv_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = [df for df in cv_frames if df is not None and not df.empty]
    if not filtered:
        return pd.DataFrame(), pd.DataFrame()
    merged = pd.concat(filtered, ignore_index=True)
    aggregated = (
        merged.groupby("fold")[["mse", "rmse", "mae", "r2", "spearman", "pearson"]]
        .mean()
        .reset_index()
    )
    aggregated = aggregated.sort_values("fold").reset_index(drop=True)
    merged = merged.sort_values(["fold", "chunk_index"]).reset_index(drop=True)
    return aggregated, merged


def build_output_dirs(root: Path, base: str, model: str, overwrite: bool) -> tuple[Path, Path]:
    run_dir = root / f"{base}_{model}"
    model_dir = run_dir / "models" / model
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory {run_dir} already exists. Use --overwrite to replace it."
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, model_dir


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_run_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_run_config(path: Path, base_config: dict, manifest: list[dict]) -> None:
    config = dict(base_config)
    config["chunk_index"] = None
    if manifest:
        config["aggregated_from_chunks"] = manifest
        chunk_total = manifest[0]["chunk_total"]
        config["chunk_total"] = chunk_total
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def combine_chunks_for_model(
    records: list[ChunkRecord],
    output_root: Path,
    include_predictions: bool,
    overwrite: bool,
) -> None:
    if not records:
        return
    base = records[0].base_name
    model = records[0].model
    chunk_total = records[0].chunk_total
    records = sorted(records, key=lambda rec: rec.chunk_index)
    chunk_manifest = []
    metrics_frames = []
    metrics_with_chunk_frames = []
    selected_frames = []
    predictions_frames = []
    cv_frames = []
    base_config: dict = {}

    for record in records:
        chunk_manifest.append(
            {
                "run_name": record.run_name,
                "chunk_index": record.chunk_index,
                "chunk_total": record.chunk_total,
                "run_dir": str(record.run_dir.resolve()),
            }
        )
        if record.chunk_total != chunk_total:
            raise ValueError(
                f"Inconsistent chunk_total for {record.run_name}: "
                f"{record.chunk_total} vs expected {chunk_total}"
            )
        if not base_config:
            base_config = load_run_config(record.run_dir / "run_configuration.json")

        # Selected genes
        selected_path = record.run_dir / "selected_genes.csv"
        if selected_path.exists():
            selected_df = load_dataframe(
                selected_path,
                {"chunk_index": record.chunk_index},
            )
            selected_frames.append(selected_df)

        # Metrics per gene
        metrics_path = record.model_dir / "metrics_per_gene.csv"
        if metrics_path.exists():
            metrics_df = load_dataframe(
                metrics_path,
                {"chunk_index": record.chunk_index},
            )
            metrics_with_chunk_frames.append(metrics_df)
            metrics_frames.append(metrics_df.drop(columns=["chunk_index"]))
        else:
            print(f"[WARN] Missing metrics_per_gene.csv in {metrics_path}", file=sys.stderr)

        # CV metrics
        cv_path = record.model_dir / "metrics_cv.csv"
        if cv_path.exists():
            cv_df = load_dataframe(
                cv_path,
                {"chunk_index": record.chunk_index},
            )
            cv_frames.append(cv_df)

        # Predictions
        if include_predictions:
            predictions_path = record.model_dir / "predictions_raw.csv"
            if predictions_path.exists():
                predictions_df = load_dataframe(
                    predictions_path,
                    {"chunk_index": record.chunk_index},
                )
                predictions_frames.append(predictions_df)

    run_dir, model_dir = build_output_dirs(output_root, base, model, overwrite)

    selected_all = concat_and_dedupe(selected_frames, subset=["gene_name", "gene_id"])
    selected_by_chunk = concat_and_dedupe(selected_frames)
    metrics_per_gene = concat_and_dedupe(metrics_frames, subset=["gene", "split"])
    metrics_per_gene_by_chunk = concat_and_dedupe(metrics_with_chunk_frames, subset=None)
    metrics_aggregate = compute_split_means(metrics_per_gene)
    cv_aggregate, cv_by_chunk = combine_cv_metrics(cv_frames)
    predictions_combined = concat_and_dedupe(predictions_frames, subset=["split", "cell_id", "gene"]) if predictions_frames else pd.DataFrame()

    # Write outputs
    write_dataframe(selected_all.sort_values("gene_name"), run_dir / "selected_genes.csv")
    write_dataframe(selected_by_chunk.sort_values(["chunk_index", "gene_name"]), run_dir / "selected_genes_by_chunk.csv")
    write_dataframe(metrics_per_gene.sort_values(["split", "gene"]), model_dir / "metrics_per_gene.csv")
    write_dataframe(metrics_per_gene_by_chunk.sort_values(["chunk_index", "split", "gene"]), model_dir / "metrics_per_gene_by_chunk.csv")
    write_dataframe(metrics_aggregate, model_dir / "metrics_aggregate.csv")
    if not cv_aggregate.empty:
        write_dataframe(cv_aggregate, model_dir / "metrics_cv.csv")
    if not cv_by_chunk.empty:
        write_dataframe(cv_by_chunk, model_dir / "metrics_cv_by_chunk.csv")
    if include_predictions and not predictions_combined.empty:
        write_dataframe(
            predictions_combined.sort_values(["split", "cell_id", "gene"]),
            model_dir / "predictions_raw.csv",
        )
        write_dataframe(
            concat_and_dedupe(predictions_frames),
            model_dir / "predictions_by_chunk.csv",
        )

    manifest_path = run_dir / "chunk_manifest.json"
    manifest_path.write_text(json.dumps(chunk_manifest, indent=2), encoding="utf-8")
    write_run_config(run_dir / "run_configuration.json", base_config, chunk_manifest)

    print(
        f"[OK] Combined {len(records)} chunks for run '{base}' model '{model}' "
        f"-> {model_dir}"
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine results from chunked runs.")
    parser.add_argument(
        "--input-root",
        default="output/results",
        type=Path,
        help="Directory containing chunked run outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Destination for combined outputs (defaults to input-root).",
    )
    parser.add_argument(
        "--run-prefix",
        action="append",
        dest="run_prefixes",
        help="Base run name to combine (can be specified multiple times). "
        "If omitted, all detected chunked runs are processed.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model identifier to combine (e.g., mlp, transformer). Can be provided multiple times.",
    )
    parser.add_argument(
        "--include-predictions",
        action="store_true",
        help="Concatenate predictions_raw.csv (can be large).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing combined output directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List combinations without writing outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    input_root: Path = args.input_root.resolve()
    output_root: Path = (args.output_root or input_root).resolve()

    if not input_root.exists():
        print(f"[ERROR] Input root {input_root} does not exist.", file=sys.stderr)
        return 1

    run_prefixes = set(args.run_prefixes) if args.run_prefixes else None
    models = set(args.models) if args.models else None

    grouped = discover_chunks(input_root, run_prefixes, models)
    if not grouped:
        print("[INFO] No chunked runs detected with the provided filters.")
        return 0

    for (base, model), records in sorted(grouped.items()):
        print(f"[INFO] Found {len(records)} chunk folders for run '{base}' model '{model}'.")
        if args.dry_run:
            continue
        try:
            combine_chunks_for_model(
                records=records,
                output_root=output_root,
                include_predictions=args.include_predictions,
                overwrite=args.overwrite,
            )
        except FileExistsError as exc:
            print(f"[SKIP] {exc}", file=sys.stderr)
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[ERROR] Failed to combine chunks for run '{base}' model '{model}': {exc}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

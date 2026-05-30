#!/usr/bin/env python3
"""List duplicate W&B runs by matching test_pearson, val_spearman, and train_r2.

Two runs are considered duplicates when all three summary metrics match to 6
decimal places AND they share the same model name and dataset.

Usage:
    python scripts/list_duplicate_runs.py --project SPEAR_v2
    python scripts/list_duplicate_runs.py --project SPEAR_v2 --entity myteam
    python scripts/list_duplicate_runs.py --project SPEAR_v2 --by-fingerprint
    python scripts/list_duplicate_runs.py --project SPEAR_v2 --json > dupes.json
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Optional


def _round_metric(value, decimals: int = 6) -> Optional[float]:
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def fetch_runs(entity: Optional[str], project: str):
    try:
        import wandb
    except ImportError:
        print("Error: wandb is not installed. Run: pip install wandb", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api(timeout=30)
    path = f"{entity}/{project}" if entity else project
    print(f"Fetching runs from {path} ...", file=sys.stderr)
    return list(api.runs(path, per_page=1000))


def find_duplicates_by_metrics(runs, decimals: int = 6):
    """Group runs by (model, dataset, test_pearson, val_spearman, train_r2)."""
    groups: dict = defaultdict(list)

    for run in runs:
        model = (
            run.config.get("model")
            or run.config.get("training", {}).get("model")
            or "unknown"
        )
        dataset = run.config.get("dataset") or "unknown"

        tp = _round_metric(run.summary.get("test_pearson"), decimals)
        vs = _round_metric(run.summary.get("val_spearman"), decimals)
        tr = _round_metric(run.summary.get("train_r2"), decimals)

        if tp is None or vs is None or tr is None:
            continue

        key = (model, dataset, tp, vs, tr)
        groups[key].append(
            {
                "id": run.id,
                "name": run.name,
                "url": run.url,
                "state": run.state,
                "created_at": str(run.created_at),
            }
        )

    return {str(k): v for k, v in groups.items() if len(v) > 1}


def find_duplicates_by_fingerprint(runs):
    """Group runs by run_fingerprint config field."""
    groups: dict = defaultdict(list)

    for run in runs:
        fp = run.config.get("run_fingerprint")
        if not fp:
            continue
        groups[fp].append(
            {
                "id": run.id,
                "name": run.name,
                "url": run.url,
                "state": run.state,
                "created_at": str(run.created_at),
            }
        )

    return {k: v for k, v in groups.items() if len(v) > 1}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--entity", help="W&B entity/team name")
    parser.add_argument(
        "--by-fingerprint",
        action="store_true",
        help="Match by run_fingerprint config field instead of metrics",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Decimal places for metric comparison (default 6)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Output as JSON"
    )
    args = parser.parse_args()

    runs = fetch_runs(args.entity, args.project)
    print(f"Fetched {len(runs)} runs.", file=sys.stderr)

    if args.by_fingerprint:
        dupes = find_duplicates_by_fingerprint(runs)
        label = "run_fingerprint"
    else:
        dupes = find_duplicates_by_metrics(runs, decimals=args.decimals)
        label = "(model, dataset, test_pearson, val_spearman, train_r2)"

    if args.json_output:
        print(json.dumps(dupes, indent=2))
        return

    if not dupes:
        print(f"No duplicate runs found (matched by {label}).")
        return

    total_duplicate_runs = sum(len(v) for v in dupes.values())
    print(
        f"\nFound {len(dupes)} duplicate group(s) covering {total_duplicate_runs} runs (matched by {label}):\n"
    )

    for key, group_runs in sorted(dupes.items()):
        print(f"  Key: {key}")
        for r in group_runs:
            print(
                f"    [{r['state']:10s}] {r['name']:40s}  {r['created_at']}  {r['url']}"
            )
        print()


if __name__ == "__main__":
    main()

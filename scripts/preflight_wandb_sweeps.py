#!/usr/bin/env python3
"""Validate SPEAR W&B sweep YAMLs before launching expensive jobs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_RUN_COUNTS: dict[str, int] = {
    "Q0": 1,
    "Q1": 8,
    "Q2": 24,
    "Q3": 12,
    "Q4": 14,
    "Q5": 30,
    "Q6": 36,
}
CORE_SWEEPS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
KNOWN_MODELS: set[str] = {
    "catboost",
    "cnn",
    "dcn",
    "elastic_net",
    "extra_trees",
    "graph",
    "hist_gradient_boosting",
    "lasso",
    "lstm",
    "mlp",
    "ols",
    "random_forest",
    "resnet",
    "ridge",
    "rnn",
    "svr",
    "transformer",
    "xgboost",
}
REQUIRED_COMMAND_FLAGS: tuple[str, ...] = (
    "--wandb-group",
    "--wandb-tags",
    "--atac-path",
    "--rna-path",
    "--gene-manifest",
    "--cache-dir",
)
FILE_FLAGS: tuple[str, ...] = ("--atac-path", "--rna-path", "--gene-manifest")
DIRECTORY_FLAGS: tuple[str, ...] = ("--cache-dir",)


@dataclass(frozen=True)
class SweepIssue:
    sweep: str
    severity: str
    message: str


@dataclass(frozen=True)
class SweepSummary:
    sweep: str
    path: str
    method: str
    expected_runs: int
    group: str
    tags: tuple[str, ...]
    models: tuple[str, ...]
    command_paths: dict[str, str]
    manifest_gene_count: int | None


def sweep_name_from_path(path: Path) -> str:
    match = re.match(r"(Q\d+)", path.name)
    return match.group(1) if match else path.stem


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return content


def command_tokens(sweep: dict[str, Any]) -> list[str]:
    command = sweep.get("command")
    if not isinstance(command, list):
        return []
    return [str(token) for token in command]


def flag_value(tokens: list[str], flag: str) -> str | None:
    try:
        index = tokens.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(tokens):
        return None
    value = tokens[index + 1]
    return None if value.startswith("--") else value


def flag_values(tokens: list[str], flag: str) -> tuple[str, ...]:
    try:
        index = tokens.index(flag)
    except ValueError:
        return ()
    values: list[str] = []
    for token in tokens[index + 1 :]:
        if token.startswith("--"):
            break
        values.append(token)
    return tuple(values)


def grid_run_count(parameters: dict[str, Any]) -> int | None:
    count = 1
    for parameter in parameters.values():
        if not isinstance(parameter, dict):
            return None
        if "values" in parameter:
            values = parameter["values"]
            if not isinstance(values, list) or not values:
                return None
            count *= len(values)
        elif "value" in parameter:
            continue
        else:
            return None
    return count


def expected_run_count(
    sweep_name: str,
    sweep: dict[str, Any],
    expected_counts: dict[str, int] | None = None,
) -> int:
    if "run_cap" in sweep:
        return int(sweep["run_cap"])

    method = str(sweep.get("method", "")).lower()
    parameters = sweep.get("parameters") or {}
    if method == "grid" and isinstance(parameters, dict):
        count = grid_run_count(parameters)
        if count is not None:
            return count

    counts = EXPECTED_RUN_COUNTS if expected_counts is None else expected_counts
    if sweep_name in counts:
        return counts[sweep_name]
    raise ValueError(
        f"Cannot infer run count for {sweep_name}; add run_cap or expected count"
    )


def model_values(sweep: dict[str, Any], tokens: list[str]) -> tuple[str, ...]:
    model_names_param = sweep.get("parameters", {}).get("model_names", {})
    if isinstance(model_names_param, dict) and "values" in model_names_param:
        models: list[str] = []
        for value in model_names_param.get("values", []):
            if isinstance(value, list):
                models.extend(str(item) for item in value)
            else:
                models.append(str(value))
        return tuple(models)

    model_arg = flag_value(tokens, "--models")
    if model_arg == "${model}":
        model_param = sweep.get("parameters", {}).get("model", {})
        values = model_param.get("values", []) if isinstance(model_param, dict) else []
        return tuple(str(value) for value in values)
    if model_arg:
        return tuple(model_arg.split())
    return ()


def has_multi_model_sweep_values(sweep: dict[str, Any]) -> bool:
    parameters = sweep.get("parameters", {})
    if not isinstance(parameters, dict):
        return False
    for parameter_name in ("model_names", "models"):
        parameter = parameters.get(parameter_name, {})
        if not isinstance(parameter, dict):
            continue
        for value_key in ("value", "values"):
            if value_key not in parameter:
                continue
            value = parameter[value_key]
            candidates = (
                value if value_key == "values" and isinstance(value, list) else [value]
            )
            for candidate in candidates:
                if isinstance(candidate, list) and len(candidate) != 1:
                    return True
    return False


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def count_manifest_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = True
        if has_header:
            reader = csv.DictReader(handle)
            return sum(1 for row in reader if any(row.values()))
        return sum(1 for line in handle if line.strip())


def expected_manifest_count_from_name(path: Path) -> int | None:
    match = re.search(r"_(\d+)\.csv$", path.name)
    return int(match.group(1)) if match else None


def validate_sweep_file(
    path: Path,
    *,
    base_dir: Path,
    expected_counts: dict[str, int] | None = None,
) -> tuple[SweepSummary | None, list[SweepIssue]]:
    sweep_name = sweep_name_from_path(path)
    issues: list[SweepIssue] = []
    try:
        sweep = load_yaml(path)
    except Exception as exc:
        return None, [SweepIssue(sweep_name, "error", f"Cannot parse YAML: {exc}")]

    tokens = command_tokens(sweep)
    if not tokens:
        issues.append(SweepIssue(sweep_name, "error", "Missing command list"))

    for flag in REQUIRED_COMMAND_FLAGS:
        if flag not in tokens:
            issues.append(SweepIssue(sweep_name, "error", f"Missing {flag}"))

    group = flag_value(tokens, "--wandb-group") or ""
    if not group:
        issues.append(SweepIssue(sweep_name, "error", "Missing W&B group value"))

    tags = flag_values(tokens, "--wandb-tags")
    if not tags:
        issues.append(SweepIssue(sweep_name, "error", "Missing W&B tags"))
    if sweep_name.startswith("Q") and not any(
        tag.startswith(sweep_name) for tag in tags
    ):
        issues.append(
            SweepIssue(
                sweep_name,
                "error",
                f"W&B tags must include a {sweep_name}-prefixed tag",
            )
        )

    try:
        runs = expected_run_count(sweep_name, sweep, expected_counts)
    except Exception as exc:
        runs = 0
        issues.append(SweepIssue(sweep_name, "error", str(exc)))

    expected_map = EXPECTED_RUN_COUNTS if expected_counts is None else expected_counts
    expected = expected_map.get(sweep_name)
    if expected is not None and runs != expected:
        issues.append(
            SweepIssue(
                sweep_name,
                "error",
                f"Expected {expected} runs but sweep resolves to {runs}",
            )
        )

    models = model_values(sweep, tokens)
    if "--wandb-sweep" in tokens and "${model}" in tokens:
        issues.append(
            SweepIssue(
                sweep_name,
                "error",
                "W&B sweep commands must not use ${model}; use scalar model_names "
                "sweep values and a valid one-model --models placeholder instead.",
            )
        )
    if "--wandb-sweep" in tokens:
        command_model_values = flag_values(tokens, "--models")
        if len(command_model_values) != 1:
            issues.append(
                SweepIssue(
                    sweep_name,
                    "error",
                    "W&B sweep command must pass exactly one --models placeholder value.",
                )
            )
    if has_multi_model_sweep_values(sweep):
        issues.append(
            SweepIssue(
                sweep_name,
                "error",
                "W&B sweep model values must select exactly one model per run.",
            )
        )
    unknown_models = sorted(set(models) - KNOWN_MODELS)
    if unknown_models:
        issues.append(
            SweepIssue(
                sweep_name,
                "error",
                f"Unknown model names: {', '.join(unknown_models)}",
            )
        )

    command_paths: dict[str, str] = {}
    manifest_gene_count: int | None = None
    for flag in FILE_FLAGS:
        value = flag_value(tokens, flag)
        if value is None:
            continue
        resolved = resolve_path(base_dir, value)
        command_paths[flag] = value
        if not resolved.is_file():
            issues.append(SweepIssue(sweep_name, "error", f"{flag} missing: {value}"))
            continue
        if flag == "--gene-manifest":
            manifest_gene_count = count_manifest_rows(resolved)
            expected_manifest_count = expected_manifest_count_from_name(resolved)
            if (
                expected_manifest_count is not None
                and manifest_gene_count != expected_manifest_count
            ):
                issues.append(
                    SweepIssue(
                        sweep_name,
                        "error",
                        "Manifest row count mismatch: "
                        f"{value} name implies {expected_manifest_count}, "
                        f"found {manifest_gene_count}",
                    )
                )

    for flag in DIRECTORY_FLAGS:
        value = flag_value(tokens, flag)
        if value is None:
            continue
        resolved = resolve_path(base_dir, value)
        command_paths[flag] = value
        if resolved.exists() and not resolved.is_dir():
            issues.append(
                SweepIssue(sweep_name, "error", f"{flag} is not a directory: {value}")
            )
        elif not resolved.exists() and not resolved.parent.is_dir():
            issues.append(
                SweepIssue(
                    sweep_name,
                    "error",
                    f"{flag} parent directory missing: {resolved.parent}",
                )
            )

    return (
        SweepSummary(
            sweep=sweep_name,
            path=path.as_posix(),
            method=str(sweep.get("method", "")),
            expected_runs=runs,
            group=group,
            tags=tags,
            models=models,
            command_paths=command_paths,
            manifest_gene_count=manifest_gene_count,
        ),
        issues,
    )


def run_preflight(
    sweep_paths: list[Path],
    *,
    base_dir: Path,
    expected_counts: dict[str, int] | None = None,
) -> tuple[list[SweepSummary], list[SweepIssue]]:
    summaries: list[SweepSummary] = []
    issues: list[SweepIssue] = []
    for path in sweep_paths:
        summary, sweep_issues = validate_sweep_file(
            path, base_dir=base_dir, expected_counts=expected_counts
        )
        if summary is not None:
            summaries.append(summary)
        issues.extend(sweep_issues)

    expected = EXPECTED_RUN_COUNTS if expected_counts is None else expected_counts
    by_name = {summary.sweep: summary for summary in summaries}
    missing = sorted(set(expected) - set(by_name))
    for sweep_name in missing:
        issues.append(SweepIssue(sweep_name, "error", "Expected sweep YAML missing"))

    core_total = sum(
        by_name[sweep_name].expected_runs
        for sweep_name in CORE_SWEEPS
        if sweep_name in by_name
    )
    if all(sweep_name in by_name for sweep_name in CORE_SWEEPS) and core_total != 124:
        issues.append(
            SweepIssue("Q1-Q6", "error", f"Expected 124 core runs, found {core_total}")
        )

    all_total = sum(summary.expected_runs for summary in summaries)
    if (
        set(EXPECTED_RUN_COUNTS).issubset(expected)
        and set(expected).issubset(by_name)
        and all_total != 125
    ):
        issues.append(
            SweepIssue("Q0-Q6", "error", f"Expected 125 total runs, found {all_total}")
        )

    return summaries, issues


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SPEAR W&B sweep YAMLs and expected run counts."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve relative paths.",
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("jobs/wandb_sweeps"),
        help="Directory containing Q0-Q6 W&B sweep YAML files.",
    )
    parser.add_argument(
        "--sweep-yaml",
        action="append",
        type=Path,
        help="Specific sweep YAML to validate. May be provided multiple times.",
    )
    parser.add_argument(
        "--expected-count",
        action="append",
        default=[],
        help="Expected run count as SWEEP_NAME=COUNT. May be provided multiple times.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write a JSON preflight summary.",
    )
    return parser.parse_args(argv)


def parse_expected_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --expected-count value: {value}")
        name, count = value.split("=", 1)
        counts[name.strip()] = int(count)
    return counts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = args.base_dir.expanduser().resolve()
    if args.sweep_yaml:
        sweep_paths = [
            resolve_path(base_dir, path.as_posix()) for path in args.sweep_yaml
        ]
        expected_counts = parse_expected_counts(args.expected_count)
    else:
        sweep_dir = resolve_path(base_dir, args.sweep_dir.as_posix())
        sweep_paths = sorted(sweep_dir.glob("Q*.yaml"))
        expected_counts = EXPECTED_RUN_COUNTS

    summaries, issues = run_preflight(
        sweep_paths, base_dir=base_dir, expected_counts=expected_counts
    )
    for summary in summaries:
        print(
            f"{summary.sweep}: {summary.expected_runs} runs, "
            f"method={summary.method}, group={summary.group}, "
            f"manifest_genes={summary.manifest_gene_count}"
        )

    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    for issue in issues:
        prefix = "ERROR" if issue.severity == "error" else "WARN"
        print(f"{prefix} [{issue.sweep}] {issue.message}", file=sys.stderr)

    if args.json_out:
        payload = {
            "summaries": [asdict(summary) for summary in summaries],
            "issues": [asdict(issue) for issue in issues],
            "error_count": len(errors),
            "warning_count": len(warnings),
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if errors:
        print(f"Preflight failed: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1

    print(f"Preflight passed: {len(summaries)} sweep(s), {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

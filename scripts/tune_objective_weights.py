#!/usr/bin/env python3
"""Simple hyperparameter tuning for optimization.objective_weights.

This script runs scripts/run_simulation.py multiple times. Before each run, it
updates optimization.objective_weights in config/simulation_config.yaml, then
collects objective score and key KPIs from the produced *_stats.json file.

Important: The original config file content is restored at the end, even if a
run fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


DEFAULT_WEIGHT_KEYS = [
    "pressure",
    "deferral_priority",
    "bottleneck",
    "utilization",
]


@dataclass
class RunResult:
    run_index: int
    return_code: int
    weights: Dict[str, float]
    stats_file: Optional[str]
    objective_score: Optional[float]
    completion_rate: Optional[float]
    total_drained_tasks: Optional[float]
    total_deferred_tasks: Optional[float]
    total_dropped_tasks: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune optimization.objective_weights by repeated simulation runs."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of tuning runs to execute (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampled combinations.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/simulation_config.yaml"),
        help="Path to simulation config YAML.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_simulation.py"),
        help="Path to simulation runner script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/simulation_outputs"),
        help="Directory where run_simulation outputs *_stats.json.",
    )
    return parser.parse_args()


def load_config_yaml(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected YAML structure in {config_path}")
    return data


def get_base_weights(config_dict: Dict) -> Dict[str, float]:
    opt = config_dict.get("optimization", {})
    weights = opt.get("objective_weights", {})
    if not isinstance(weights, dict):
        raise ValueError("optimization.objective_weights is missing or not a mapping")

    missing = [k for k in DEFAULT_WEIGHT_KEYS if k not in weights]
    if missing:
        raise ValueError(f"Missing objective weight keys in config: {missing}")

    return {k: float(weights[k]) for k in DEFAULT_WEIGHT_KEYS}


def replace_optimization_objective_weights_block(
    config_text: str,
    new_weights: Dict[str, float],
) -> str:
    """Replace the optimization.objective_weights block in-place.

    This avoids rewriting the full YAML file and preserves the surrounding
    comments and formatting as much as possible.
    """
    lines = config_text.splitlines(keepends=True)

    # Find top-level optimization section.
    opt_start = None
    top_level_key_re = re.compile(r"^[A-Za-z0-9_\-]+:\s*(#.*)?$")
    for i, line in enumerate(lines):
        if line.startswith("optimization:"):
            opt_start = i
            break
    if opt_start is None:
        raise ValueError("Could not find top-level 'optimization:' section")

    opt_end = len(lines)
    for i in range(opt_start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not lines[i].startswith(" ") and top_level_key_re.match(stripped):
            opt_end = i
            break

    # Find '  objective_weights:' inside optimization section.
    ow_start = None
    for i in range(opt_start + 1, opt_end):
        if re.match(r"^\s{2}objective_weights:\s*(#.*)?$", lines[i].rstrip("\n")):
            ow_start = i
            break
    if ow_start is None:
        raise ValueError("Could not find 'optimization.objective_weights' block")

    # Objective weight entries are expected to be indented with 4 spaces.
    ow_end = ow_start + 1
    while ow_end < opt_end:
        line = lines[ow_end]
        if line.strip() == "":
            ow_end += 1
            continue
        if line.startswith("    "):
            ow_end += 1
            continue
        break

    new_block = [lines[ow_start]]
    for key in DEFAULT_WEIGHT_KEYS:
        value = float(new_weights[key])
        new_block.append(f"    {key}: {value:.4f}\n")

    updated = lines[:ow_start] + new_block + lines[ow_end:]
    return "".join(updated)


def sample_random_combo(rng: random.Random, weight_sum: float) -> Dict[str, float]:
    # Dirichlet(alpha=1) using gamma samples, no numpy dependency.
    raw = [rng.gammavariate(1.0, 1.0) for _ in DEFAULT_WEIGHT_KEYS]
    total = sum(raw)
    combo = {}
    for key, val in zip(DEFAULT_WEIGHT_KEYS, raw):
        combo[key] = round((val / total) * weight_sum, 4)
    return combo


def build_weight_combinations(base: Dict[str, float], runs: int, seed: int) -> List[Dict[str, float]]:
    if runs <= 0:
        return []

    rng = random.Random(seed)
    target_sum = sum(base.values())
    combos: List[Dict[str, float]] = []
    seen: set[Tuple[float, ...]] = set()

    def add_combo(c: Dict[str, float]) -> None:
        signature = tuple(c[k] for k in DEFAULT_WEIGHT_KEYS)
        if signature not in seen and len(combos) < runs:
            seen.add(signature)
            combos.append(c)

    # 1) Baseline as first run.
    add_combo({k: round(float(base[k]), 4) for k in DEFAULT_WEIGHT_KEYS})

    # 2) Focused runs: one objective emphasized each.
    low = target_sum * 0.1
    high = target_sum - low * (len(DEFAULT_WEIGHT_KEYS) - 1)
    for key in DEFAULT_WEIGHT_KEYS:
        c = {k: round(low, 4) for k in DEFAULT_WEIGHT_KEYS}
        c[key] = round(high, 4)
        add_combo(c)

    # 3) Fill remaining slots with random samples.
    while len(combos) < runs:
        add_combo(sample_random_combo(rng, target_sum))

    return combos


def find_newest_stats_file(output_dir: Path, before_snapshot: set[Path]) -> Optional[Path]:
    after = set(output_dir.glob("sim_*_stats.json"))
    created = [p for p in after - before_snapshot if p.is_file()]
    if created:
        return max(created, key=lambda p: p.stat().st_mtime)

    # Fallback if no clean diff is available.
    all_stats = [p for p in output_dir.glob("sim_*_stats.json") if p.is_file()]
    if not all_stats:
        return None
    return max(all_stats, key=lambda p: p.stat().st_mtime)


def parse_stats(stats_file: Optional[Path]) -> Dict[str, Optional[float]]:
    if stats_file is None or not stats_file.exists():
        return {
            "objective_score": None,
            "completion_rate": None,
            "total_drained_tasks": None,
            "total_deferred_tasks": None,
            "total_dropped_tasks": None,
        }

    with stats_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    drain_stats = data.get("drain_stats", {}) if isinstance(data, dict) else {}
    return {
        "objective_score": data.get("objective_score"),
        "completion_rate": data.get("completion_rate"),
        "total_drained_tasks": drain_stats.get("total_drained_tasks"),
        "total_deferred_tasks": drain_stats.get("total_deferred_tasks"),
        "total_dropped_tasks": drain_stats.get("total_dropped_tasks"),
    }


def write_results(output_dir: Path, results: List[RunResult]) -> Tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"objective_weight_tuning_{ts}.csv"
    json_path = output_dir / f"objective_weight_tuning_{ts}.json"

    fieldnames = [
        "run_index",
        "return_code",
        *DEFAULT_WEIGHT_KEYS,
        "objective_score",
        "completion_rate",
        "total_drained_tasks",
        "total_deferred_tasks",
        "total_dropped_tasks",
        "stats_file",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "run_index": row.run_index,
                    "return_code": row.return_code,
                    "pressure": row.weights["pressure"],
                    "deferral_priority": row.weights["deferral_priority"],
                    "bottleneck": row.weights["bottleneck"],
                    "utilization": row.weights["utilization"],
                    "objective_score": row.objective_score,
                    "completion_rate": row.completion_rate,
                    "total_drained_tasks": row.total_drained_tasks,
                    "total_deferred_tasks": row.total_deferred_tasks,
                    "total_dropped_tasks": row.total_dropped_tasks,
                    "stats_file": row.stats_file,
                }
            )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "run_index": r.run_index,
                    "return_code": r.return_code,
                    "weights": r.weights,
                    "objective_score": r.objective_score,
                    "completion_rate": r.completion_rate,
                    "total_drained_tasks": r.total_drained_tasks,
                    "total_deferred_tasks": r.total_deferred_tasks,
                    "total_dropped_tasks": r.total_dropped_tasks,
                    "stats_file": r.stats_file,
                }
                for r in results
            ],
            f,
            indent=2,
        )

    return csv_path, json_path


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = (project_root / args.config).resolve()
    runner_path = (project_root / args.runner).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_path}")

    original_text = config_path.read_text(encoding="utf-8")
    base_config = load_config_yaml(config_path)
    base_weights = get_base_weights(base_config)
    combinations = build_weight_combinations(base_weights, args.runs, args.seed)

    print(f"Tuning runs planned: {len(combinations)}")
    print(f"Config: {config_path}")
    print(f"Runner: {runner_path}")
    print(f"Output dir: {output_dir}")

    results: List[RunResult] = []

    try:
        for idx, weights in enumerate(combinations, start=1):
            print("\n" + "=" * 80)
            print(f"Run {idx}/{len(combinations)}")
            print("Weights:", weights)

            updated_text = replace_optimization_objective_weights_block(original_text, weights)
            config_path.write_text(updated_text, encoding="utf-8")

            before_snapshot = set(output_dir.glob("sim_*_stats.json"))
            completed = subprocess.run(
                [sys.executable, str(runner_path)],
                cwd=str(project_root),
                check=False,
            )

            stats_file = find_newest_stats_file(output_dir, before_snapshot)
            stats = parse_stats(stats_file)

            result = RunResult(
                run_index=idx,
                return_code=completed.returncode,
                weights=weights,
                stats_file=str(stats_file) if stats_file else None,
                objective_score=stats["objective_score"],
                completion_rate=stats["completion_rate"],
                total_drained_tasks=stats["total_drained_tasks"],
                total_deferred_tasks=stats["total_deferred_tasks"],
                total_dropped_tasks=stats["total_dropped_tasks"],
            )
            results.append(result)

            print(
                "Result:",
                {
                    "return_code": result.return_code,
                    "objective_score": result.objective_score,
                    "completion_rate": result.completion_rate,
                    "stats_file": result.stats_file,
                },
            )

    finally:
        config_path.write_text(original_text, encoding="utf-8")
        print("\nOriginal config restored.")

    csv_path, json_path = write_results(output_dir, results)
    print("\nSaved tuning summary:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")

    successful = sum(1 for r in results if r.return_code == 0)
    print(f"Successful runs: {successful}/{len(results)}")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

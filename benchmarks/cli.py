"""Benchmark runner CLI.

python -m benchmarks --profile quick
python -m benchmarks --profile paper
python -m benchmarks --cores-sweep
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import cpu_count
from pathlib import Path
from statistics import mean, median
from typing import Any

from .cases import (
    BENCHMARK_CASES,
    BENCHMARK_SCENARIOS,
    BenchmarkCase,
    Scenario,
    parse_names,
)
from .engines import ENGINE_NAMES, Signature, run_engine

PROFILES: dict[str, dict[str, str]] = {
    "quick": {
        "cases": "2kp50",
        "scenarios": "augmecon_r,parallel_default",
        "repeats": "1",
    },
    "paper": {
        "cases": "all",
        "scenarios": "augmecon,augmecon_2,augmecon_r,parallel_default",
        "repeats": "3",
    },
    "full": {"cases": "all", "scenarios": "all", "repeats": "1"},
}

CORES_SWEEP = tuple(range(2, 49, 2))
CORES_SWEEP_CASES = ("3kp40", "3kp50", "4kp40", "4kp50")
CORES_SWEEP_SCENARIO = "parallel_default"

PARALLEL_ENGINES = {"pyaugmecon"}


@dataclass(frozen=True, slots=True)
class Job:
    engine: str
    solver: str
    case: BenchmarkCase
    scenario: Scenario
    workers: int
    repeat: int


def _parse_solver_opt(values: list[str]) -> dict[str, Any]:
    """Parse `KEY=VALUE` strings, coercing bool/int/float when possible."""
    out: dict[str, Any] = {}
    for item in values:
        if "=" not in item or not item.split("=", 1)[0].strip():
            raise ValueError(f"Invalid --solver-opt {item!r}. Use KEY=VALUE.")
        key, raw = (s.strip() for s in item.split("=", 1))
        if raw.lower() in {"true", "false"}:
            out[key] = raw.lower() == "true"
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
    return out


def _build_jobs(
    engines: list[str],
    cases: list[BenchmarkCase],
    scenarios: list[Scenario],
    solvers: list[str],
    repeats: int,
    workers: int,
    cores_sweep: bool,
) -> list[Job]:
    """Cross-product of engines, cases, scenarios, solvers, and repeats."""
    if cores_sweep:
        scenario = BENCHMARK_SCENARIOS[CORES_SWEEP_SCENARIO]
        sweep_cases = [c for c in cases if c.name in CORES_SWEEP_CASES]
        return [
            Job(e, s, c, scenario, w, r)
            for s in solvers
            for e in engines
            if e in PARALLEL_ENGINES
            for c in sweep_cases
            for w in CORES_SWEEP
            for r in range(1, repeats + 1)
        ]
    return [
        Job(e, s, c, sc, workers, r)
        for s in solvers
        for e in engines
        for c in cases
        for sc in scenarios
        for r in range(1, repeats + 1)
        if not (int(sc.opts.get("workers", workers)) > 1 and e not in PARALLEL_ENGINES)  # ty: ignore[invalid-argument-type]
    ]


def _aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group runs and compute per-group runtime stats and parity flags.

    Each run dict must carry a ``signature`` key with its parity signature.
    The first signature per (engine, solver, case) triple is the baseline;
    ``parity`` is True when every run in the group matches its baseline.
    Cross-engine parity compares each engine's baseline against pyaugmecon's.
    Speedup is relative to the first (alphabetically) scenario's median
    within each (engine, solver, case) group — typically "augmecon".
    """
    baselines: dict[tuple[str, str, str], Signature] = {}
    for r in runs:
        baselines.setdefault((r["engine"], r["solver"], r["case"]), r["signature"])

    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = {}
    for r in runs:
        key = (r["engine"], r["solver"], r["case"], r["scenario"], r["workers"])
        grouped.setdefault(key, []).append(r)

    speedup_ref_med: dict[tuple[str, str, str], float] = {}
    summary: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        engine, solver, case, scenario, workers = key
        times = [r["runtime_seconds"] for r in group]
        med = round(median(times), 4)
        speedup_ref_med.setdefault((engine, solver, case), med)
        own_baseline_med = speedup_ref_med[(engine, solver, case)]
        pyaugmecon_sig = baselines.get(("pyaugmecon", solver, case))
        own_sig = baselines.get((engine, solver, case))
        parity = all(
            r["signature"] == baselines[(r["engine"], r["solver"], r["case"])]
            for r in group
        )
        cross_parity = (
            "\u2014"
            if engine == "pyaugmecon" or pyaugmecon_sig is None or own_sig is None
            else pyaugmecon_sig == own_sig
        )
        summary.append(
            {
                "engine": engine,
                "solver": solver,
                "case": case,
                "scenario": scenario,
                "workers": workers,
                "repeats": len(group),
                "mean_s": round(mean(times), 4),
                "median_s": med,
                "min_s": round(min(times), 4),
                "max_s": round(max(times), 4),
                "pareto_points": group[0]["pareto_points"],
                "parity": parity,
                "speedup": round(own_baseline_med / med, 4) if med > 0 else None,
                "xparity": cross_parity,
            }
        )
    return summary


def _format_table(rows: list[dict[str, Any]]) -> str:
    cols = (
        ("engine", "Engine"),
        ("solver", "Solver"),
        ("case", "Case"),
        ("scenario", "Scenario"),
        ("workers", "Workers"),
        ("median_s", "Median(s)"),
        ("speedup", "Speedup"),
        ("pareto_points", "Pareto"),
        ("parity", "Parity"),
        ("xparity", "vs PyAugmecon"),
    )

    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    cells = [{k: fmt(r.get(k)) for k, _ in cols} for r in rows]
    widths = {
        k: max(len(label), *(len(c[k]) for c in cells)) if cells else len(label)
        for k, label in cols
    }
    lines: list[str] = [
        "  ".join(label.ljust(widths[k]) for k, label in cols),
        "  ".join("-" * widths[k] for k, _ in cols),
    ]
    lines.extend("  ".join(c[k].ljust(widths[k]) for k, _ in cols) for c in cells)
    return "\n".join(lines)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyAugmecon benchmark runner.")
    p.add_argument(
        "--engine", default="pyaugmecon", help="Comma-separated engines or 'all'."
    )
    p.add_argument("--profile", choices=tuple(PROFILES), default="quick")
    p.add_argument("--cases", default=None, help="Override profile cases.")
    p.add_argument("--scenarios", default=None, help="Override profile scenarios.")
    p.add_argument(
        "--repeats", type=int, default=None, help="Runs per (engine, case, scenario)."
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(2, min(cpu_count(), 4)),
        help="Worker count for scenarios that don't pin one.",
    )
    p.add_argument(
        "--cores-sweep",
        action="store_true",
        help=f"Sweep workers in {CORES_SWEEP} on {CORES_SWEEP_SCENARIO} "
        f"for {','.join(CORES_SWEEP_CASES)}.",
    )
    p.add_argument(
        "--solvers",
        default="highs",
        help="Comma-separated solver families. Default: highs.",
    )
    p.add_argument(
        "--solver-opt",
        action="append",
        default=[],
        help="Solver option KEY=VALUE (repeatable).",
    )
    p.add_argument("--output", default="benchmarks/results/latest.json")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = PROFILES[args.profile]
    cases = [
        BENCHMARK_CASES[n]
        for n in parse_names(args.cases or profile["cases"], BENCHMARK_CASES, "case")
    ]
    scenarios = [
        BENCHMARK_SCENARIOS[n]
        for n in parse_names(
            args.scenarios or profile["scenarios"], BENCHMARK_SCENARIOS, "scenario"
        )
    ]
    engines = parse_names(args.engine, ENGINE_NAMES, "engine")
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    solver_opts = _parse_solver_opt(args.solver_opt)
    repeats = args.repeats or int(profile["repeats"])
    if args.workers <= 0 or repeats <= 0:
        raise ValueError("--workers and --repeats must be positive.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs/benchmarks")
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(
        engines, cases, scenarios, solvers, repeats, args.workers, args.cores_sweep
    )

    runs: list[dict[str, Any]] = []
    for j in jobs:
        run, sig = run_engine(
            j.engine,
            j.case,
            j.scenario,
            j.repeat,
            j.solver,
            solver_opts,
            log_dir,
            j.workers,
        )
        run["solver"] = j.solver
        run["signature"] = sig
        runs.append(run)

    summary = _aggregate(runs)
    for r in runs:
        r.pop("signature", None)
    output_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "profile": args.profile,
                "cores_sweep": args.cores_sweep,
                "runs": runs,
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Results: {output_path}\n")
    print(_format_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

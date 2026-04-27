"""Benchmark runner CLI.

Everything is configured from YAML.

    python -m benchmarks
    python -m benchmarks --plan benchmarks/plans/default.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from .engines import ENGINES, BenchmarkCase, RunResult, Scenario, Signature, run_engine

DEFAULT_PLAN_PATH = Path(__file__).with_name("plans") / "default.yaml"


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _run_key(run: RunResult) -> tuple[Any, ...]:
    opts = tuple(sorted((str(k), repr(v)) for k, v in run.solver_options.items()))
    return (
        run.engine,
        run.solver,
        opts,
        run.case,
        run.scenario,
        int(run.workers),
        int(run.sample),
    )


def _select(spec: Any, registry: dict[str, Any], kind: str) -> list[str]:
    names = [str(v).strip() for v in _ensure_list(spec) if str(v).strip()]
    if not names:
        raise ValueError(f"Plan field '{kind}' must contain at least one value.")
    if len(names) == 1 and names[0].lower() == "all":
        return list(registry)
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(
            f"Unknown {kind}(s): {', '.join(unknown)}. Available: {', '.join(registry)}."
        )
    return names


@dataclass(frozen=True, slots=True)
class Job:
    engine: str
    solver: str
    solver_options: dict[str, Any]
    case: BenchmarkCase
    scenario: Scenario
    workers: int
    sample: int

    @property
    def key(self) -> tuple[Any, ...]:
        opts = tuple(sorted((str(k), repr(v)) for k, v in self.solver_options.items()))
        return (
            self.engine,
            self.solver,
            opts,
            self.case.name,
            self.scenario.name,
            int(self.workers),
            int(self.sample),
        )


def _load_jobs_from_plan(path: Path) -> tuple[list[Job], Path]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark plan file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Plan YAML must be a mapping: {path}")

    cases_raw: dict[str, Any] = {}
    scenarios_raw: dict[str, Any] = {}
    for include in data.get("include") or []:
        included = yaml.safe_load((path.parent / include).read_text(encoding="utf-8"))
        cases_raw.update(included.get("cases") or {})
        scenarios_raw.update(included.get("scenarios") or {})
    cases_raw.update(data.get("cases") or {})
    scenarios_raw.update(data.get("scenarios") or {})
    if not cases_raw or not scenarios_raw:
        raise ValueError("Plan must define non-empty 'cases' and 'scenarios' mappings.")

    cases = {
        name: BenchmarkCase.from_config(name, cfg) for name, cfg in cases_raw.items()
    }
    scenarios = {
        name: Scenario(name=name, opts=dict(opts))
        for name, opts in scenarios_raw.items()
    }

    jobs: list[Job] = []
    for run_cfg in data.get("runs") or []:
        name = run_cfg.get("name", "<unnamed>")
        try:
            engines = _select(run_cfg["engine"], ENGINES, "engine")
            raw_solver = run_cfg["solver"]
            solvers = [
                str(s).strip() for s in _ensure_list(raw_solver) if str(s).strip()
            ]
            case_names = _select(run_cfg["case"], cases, "case")
            scenario_names = _select(run_cfg["scenario"], scenarios, "scenario")
            workers = [int(w) for w in _ensure_list(run_cfg["workers"])]
            samples = int(run_cfg["samples"])
        except KeyError as e:
            raise ValueError(
                f"Run '{name}' missing required field: {e.args[0]}"
            ) from None
        solver_options = dict(run_cfg.get("solver_options") or {})

        if not solvers:
            raise ValueError(
                f"Run '{name}' field 'solver' must contain at least one value."
            )
        if any(w <= 0 for w in workers):
            raise ValueError(
                f"Run '{name}' field 'workers' must contain positive integers."
            )
        if samples <= 0:
            raise ValueError(
                f"Run '{name}' field 'samples' must be a positive integer."
            )

        for engine in engines:
            for solver in solvers:
                for case_name in case_names:
                    for scenario_name in scenario_names:
                        for worker in workers:
                            if worker > 1 and not ENGINES[engine]:
                                continue
                            jobs.extend(
                                Job(
                                    engine=engine,
                                    solver=solver,
                                    solver_options=dict(solver_options),
                                    case=cases[case_name],
                                    scenario=scenarios[scenario_name],
                                    workers=worker,
                                    sample=sample,
                                )
                                for sample in range(1, samples + 1)
                            )
    return jobs, path


def _annotate_parity(runs: list[RunResult]) -> None:
    baselines: dict[tuple[str, str, str], Signature] = {}
    for run in runs:
        if run.signature is not None:
            baselines.setdefault((run.engine, run.solver, run.case), run.signature)

    for run in runs:
        own = baselines.get((run.engine, run.solver, run.case))
        sig = run.signature
        if sig is None or own is None:
            run.matches_baseline = None
        else:
            run.matches_baseline = sig == own

        if run.engine == "pyaugmecon":
            run.cross_engine_parity = None
        else:
            pyaug = baselines.get(("pyaugmecon", run.solver, run.case))
            if own is None or pyaug is None:
                run.cross_engine_parity = None
            else:
                run.cross_engine_parity = own == pyaug


def _aggregate_parity(group: list[RunResult], field: str) -> bool | str | None:
    """All-True / all-False / '?' on disagreement / None when no data."""
    vals = [getattr(r, field) for r in group if getattr(r, field) is not None]
    if not vals:
        return None
    return vals[0] if len(set(vals)) == 1 else "?"


def _aggregate(runs: list[RunResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int], list[RunResult]] = {}
    for r in runs:
        key = (r.engine, r.solver, r.case, r.scenario, r.workers)
        grouped.setdefault(key, []).append(r)

    speedup_ref: dict[tuple[str, str, str], float] = {}
    summary: list[dict[str, Any]] = []
    for (engine, solver, case, scenario, workers), group in sorted(grouped.items()):
        timed_out = any(getattr(r, "timed_out", False) for r in group)

        if timed_out:
            med = None
            mean_s = None
            min_s = None
            max_s = None
        else:
            times = [r.runtime_seconds for r in group]
            med = round(median(times), 4)
            mean_s = round(mean(times), 4)
            min_s = round(min(times), 4)
            max_s = round(max(times), 4)
            speedup_ref.setdefault((engine, solver, case), med)

        baseline = speedup_ref.get((engine, solver, case))

        first = group[0]
        xparity: bool | str | None
        if timed_out:
            xparity = None
            parity = None
        elif engine == "pyaugmecon":
            xparity = "\u2014"
            parity = _aggregate_parity(group, "matches_baseline")
        else:
            xparity = _aggregate_parity(group, "cross_engine_parity")
            parity = _aggregate_parity(group, "matches_baseline")

        summary.append(
            {
                "engine": engine,
                "solver": solver,
                "case": case,
                "scenario": scenario,
                "workers": workers,
                "samples": len(group),
                "mean_s": mean_s,
                "median_s": med,
                "min_s": min_s,
                "max_s": max_s,
                "grid_points": first.grid_points,
                "objective_count": first.objective_count,
                "pareto_points": first.pareto_points,
                "dominated_points": first.dominated_points,
                "models_solved": first.models_solved,
                "models_infeasible": first.models_infeasible,
                "hv_indicator": first.hv_indicator,
                "parity": parity,
                "speedup": round(baseline / med, 4) if med and baseline else None,
                "xparity": xparity,
                "timed_out": timed_out,
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
        ("models_solved", "Models"),
        ("models_infeasible", "Infeas"),
        ("parity", "Parity"),
        ("xparity", "vs PyAugmecon"),
    )

    def fmt(v: Any, is_timeout: bool = False) -> str:
        if is_timeout and v is None:
            return "TIMEOUT"
        if v is None:
            return "-"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    cells = [
        {
            k: fmt(
                r.get(k), is_timeout=r.get("timed_out") and k in ("median_s", "speedup")
            )
            for k, _ in cols
        }
        for r in rows
    ]
    widths = {
        k: max(len(label), *(len(c[k]) for c in cells)) if cells else len(label)
        for k, label in cols
    }
    header = "  ".join(label.ljust(widths[k]) for k, label in cols)
    sep = "  ".join("-" * widths[k] for k, _ in cols)
    body = ["  ".join(c[k].ljust(widths[k]) for k, _ in cols) for c in cells]
    return "\n".join([header, sep, *body])


def _load_existing_runs(path: Path) -> list[RunResult]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [RunResult.from_dict(r) for r in payload.get("runs", [])]


def _environment() -> dict[str, str]:
    """Reproducibility metadata (hardware, OS, Python, key versions)."""
    import platform  # noqa: PLC0415
    import sys  # noqa: PLC0415
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    def pkg(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "unknown"

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "pyaugmecon": pkg("pyaugmecon"),
        "pyomo": pkg("pyomo"),
        "highspy": pkg("highspy"),
        "gurobipy": pkg("gurobipy"),
        "augmecon-py": pkg("augmecon-py"),
    }


def _write_results(
    output_path: Path, plan_path: Path, runs: list[RunResult]
) -> list[dict[str, Any]]:
    _annotate_parity(runs)
    summary = _aggregate(runs)
    run_dicts = [r.to_dict() for r in runs]
    output_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "plan": str(plan_path),
                "environment": _environment(),
                "runs": run_dicts,
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyAugmecon benchmark runner.")
    parser.add_argument(
        "--plan", default=str(DEFAULT_PLAN_PATH), help="Path to benchmark YAML plan."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (default: results/<plan_stem>.json).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing runs in --output (default: enabled).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan_path = Path(args.plan)
    output_path = (
        Path(args.output)
        if args.output
        else Path("benchmarks/results") / f"{plan_path.stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs/benchmarks")
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs, plan_path = _load_jobs_from_plan(plan_path)
    existing = _load_existing_runs(output_path) if args.resume else []

    target_keys = {j.key for j in jobs}
    runs = [r for r in existing if _run_key(r) in target_keys]
    seen = {_run_key(r) for r in runs}
    pending = [j for j in jobs if j.key not in seen]

    if not pending:
        summary = _write_results(output_path, plan_path, runs)
        print(f"Results: {output_path}")
        print(f"Planned jobs: {len(jobs)} | resumed: {len(runs)} | pending: 0\n")
        print(_format_table(summary))
        return 0

    skip_configs = {
        (r.engine, r.solver, r.case, r.scenario, r.workers)
        for r in runs
        if getattr(r, "timed_out", False)
    }

    summary = _write_results(output_path, plan_path, runs)

    for idx, job in enumerate(pending, start=1):
        config_key = (
            job.engine,
            job.solver,
            job.case.name,
            job.scenario.name,
            job.workers,
        )
        if config_key in skip_configs:
            print(
                f"Skipping [{idx}/{len(pending)}] {job.engine}/{job.solver}/{job.case.name}/{job.scenario.name}/w{job.workers}/s{job.sample} due to previous timeout"
            )
            continue

        print(
            f"Running [{idx}/{len(pending)}] "
            f"{job.engine}/{job.solver}/{job.case.name}/{job.scenario.name}"
            f"/w{job.workers}/s{job.sample}"
        )

        result = run_engine(job, log_dir)
        runs.append(result)

        if getattr(result, "timed_out", False):
            print("  -> Timed out! Skipping remaining samples for this configuration.")
            skip_configs.add(config_key)

        summary = _write_results(output_path, plan_path, runs)

    print(f"Results: {output_path}\n")
    print(
        f"Planned jobs: {len(jobs)} | resumed: {len(jobs) - len(pending)} | ran now: {len(pending)}\n"
    )
    print(_format_table(summary))
    return 0

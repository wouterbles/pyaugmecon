"""Engine adapters for the benchmark suite.

`pyaugmecon` is this project. `augmecon-py` is the reference single-process
`MoipAugmeconR` from `../augmecon-py`, available via the `benchmarks` extra
and only meaningful for single-worker scenarios.
"""

from __future__ import annotations

import contextlib
import inspect
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from pyaugmecon import PyAugmecon
from pyaugmecon.config import PyAugmeconConfig
from pyaugmecon.example_models import _load_kp_matrices

from .cases import BenchmarkCase, Scenario


class Signature(NamedTuple):
    """Hashable, cross-engine-comparable rounded Pareto + payoff snapshot."""

    pareto: tuple[tuple[float, ...], ...]
    payoff: tuple[tuple[float, ...], ...]


ENGINE_NAMES: dict[str, str] = {
    "pyaugmecon": "This project (parallel-capable AUGMECON-R).",
    "augmecon-py": "Reference single-process MoipAugmeconR.",
}


def _signature(
    points: Sequence[Sequence[float]], payoff: np.ndarray, decimals: int = 9
) -> Signature:
    pts = sorted({tuple(round(float(v), decimals) for v in p) for p in points})
    pay = np.round(np.asarray(payoff, dtype=float), decimals).tolist()
    return Signature(
        pareto=tuple(pts),
        payoff=tuple(tuple(float(v) for v in row) for row in pay),
    )


def _run_pyaugmecon(
    case: BenchmarkCase,
    scenario: Scenario,
    repeat: int,
    solver_name: str,
    solver_opts: dict[str, Any],
    log_dir: Path,
    workers: int,
) -> tuple[dict[str, Any], Signature]:
    cfg_kwargs = {
        **case.case_opts(),
        **scenario.opts,
        "name": f"{case.name}_{scenario.name}_r{repeat}",
        "mode": "sampled",
        "solver_name": solver_name,
        "solver_options": solver_opts,
        "artifact_folder": str(log_dir),
        "write_csv": False,
        "progress_bar": False,
        "log_to_console": False,
    }
    cfg_kwargs.setdefault("workers", workers)

    runner = PyAugmecon(case.build_model(), PyAugmeconConfig(**cfg_kwargs))  # ty: ignore[invalid-argument-type]
    result = runner.solve()
    run = {
        "engine": "pyaugmecon",
        "case": case.name,
        "scenario": scenario.name,
        "workers": cfg_kwargs["workers"],
        "repeat": repeat,
        "runtime_seconds": result.runtime_seconds,
        "hv_indicator": result.hypervolume(),
        "pareto_points": result.count,
        "models_solved": result.models_solved,
        "models_infeasible": result.models_infeasible,
    }
    return run, _signature(result.points, result.payoff_table)


def _build_kp_model(case: BenchmarkCase):
    """Knapsack pyomo model in the shape augmecon-py's MoipAugmeconR expects."""
    import pyomo.environ as p  # noqa: PLC0415

    a, b, c = _load_kp_matrices(case.name)
    n_obj, n_items = case.objective_count, len(a[0])

    model = p.ConcreteModel()
    model.ITEMS = p.Set(initialize=range(n_items))
    model.DecisionVariable = p.Var(model.ITEMS, within=p.Binary)
    model.obj_list = p.ObjectiveList()
    for k in range(n_obj):
        model.obj_list.add(
            expr=sum(c[k][i] * model.DecisionVariable[i] for i in model.ITEMS),
            sense=p.maximize,
        )
    model.kp_constraints = p.ConstraintList()
    for k in range(len(b)):
        model.kp_constraints.add(  # ty: ignore[missing-argument]
            expr=sum(a[k][i] * model.DecisionVariable[i] for i in model.ITEMS)
            <= b[k][0]
        )
    model.obj_list.deactivate()
    model.undo_auto_swap_proxy = np.argsort(list(range(1, n_obj + 1)))
    return model


def _wrap_solve_for_augmecon_py(solver) -> None:
    """Force `load_solutions=False` on `solver.solve` calls.

    augmecon-py calls `solver.solve(model)` with no kwargs and inspects the
    result's termination_condition. Modern Pyomo backends (notably `highs`)
    raise `NoFeasibleSolutionError` from `solve()` before the caller ever
    sees the result. Disabling auto-load preserves augmecon-py's control flow.
    """
    original = solver.solve
    try:
        params = inspect.signature(original).parameters
    except (TypeError, ValueError):
        return
    if "load_solutions" not in params:
        return

    from pyomo.opt import SolverStatus, TerminationCondition  # noqa: PLC0415

    loadable = {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal,
    }
    infeasible_conds = {
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    }

    def patched(model, *args, **kwargs):
        kwargs.setdefault("load_solutions", False)
        result = original(model, *args, **kwargs)
        try:
            tc = result.solver.termination_condition
        except AttributeError:
            return result
        if tc in loadable:
            try:
                solver.load_vars()
            except (AttributeError, NotImplementedError):
                with contextlib.suppress(Exception):
                    model.solutions.load_from(result)
        elif tc in infeasible_conds:
            with contextlib.suppress(AttributeError):
                result.solver.status = SolverStatus.warning
        return result

    solver.solve = patched


def _build_augmecon_py_solver(solver_name: str, solver_opts: dict[str, Any]):
    """Construct a Pyomo solver compatible with augmecon-py.

    augmecon-py hard-codes gurobi with `solver_io='python'`. This honors the
    benchmark's `--solvers` choice, preferring direct backends whose
    `solve()` doesn't raise on infeasible.
    """
    import pyomo.environ as p  # noqa: PLC0415

    family = (solver_name or "gurobi").lower()
    candidates = {
        "gurobi": [("gurobi", {"solver_io": "python"}), ("gurobi_direct", {})],
        "highs": [("highs", {}), ("appsi_highs", {})],
        "xpress": [("xpress_direct", {}), ("xpress", {})],
        "cplex": [("cplex_direct", {}), ("cplex", {})],
        "cbc": [("cbc", {})],
    }.get(family, [(solver_name, {})])

    last_err: Exception | None = None
    for backend, factory_kwargs in candidates:
        try:
            solver = p.SolverFactory(backend, **factory_kwargs)
            available = getattr(solver, "available", None)
            if solver is None or (callable(available) and not available(False)):
                last_err = RuntimeError(f"backend '{backend}' unavailable")
                continue
            for k, v in solver_opts.items():
                with contextlib.suppress(Exception):
                    solver.options[k] = v
            _wrap_solve_for_augmecon_py(solver)
            return solver, backend
        except Exception as exc:  # pragma: no cover
            last_err = exc
    raise RuntimeError(
        f"No augmecon-py-compatible backend for solver family {family!r}. "
        f"Last error: {last_err}"
    )


def _run_augmecon_py(
    case: BenchmarkCase,
    scenario: Scenario,
    repeat: int,
    solver_name: str,
    solver_opts: dict[str, Any],
    _log_dir: Path,
    _workers: int,
) -> tuple[dict[str, Any], Signature]:
    if scenario.opts.get("workers") != 1:
        raise RuntimeError(
            "augmecon-py is single-process; only single-worker scenarios are meaningful."
        )
    try:
        from augmecon_py import MoipAugmeconR  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "augmecon-py engine requires the `augmecon-py` package. "
            "Install with: `uv sync --extra benchmarks`."
        ) from exc

    solver, backend = _build_augmecon_py_solver(solver_name, solver_opts)
    nadir = case.nadir_points
    # `fixed_nadirs` is 1-indexed; entries 2..n drive the grid.
    fixed_nadirs = None if nadir is None else [None, None, *nadir]

    started = time.perf_counter()
    runner = MoipAugmeconR(
        _build_kp_model(case),
        solver=solver,
        model_name=f"{case.name}_r{repeat}",
        # Use payoff-derived nadirs verbatim; default 0.8 widens by 20%.
        min_to_nadir_undercut=1.0,
        fixed_nadirs=fixed_nadirs,
    )
    # `execute` prints every solution; redirect to keep benchmark output clean.
    with contextlib.redirect_stdout(None):
        runner.execute()
    runtime = round(time.perf_counter() - started, 2)

    pareto = [
        [float(v) for v in s.objective_values] for s in runner.pareto_front.values()
    ]
    payoff = runner.payoff_table[1:, 1:]  # drop unused 0-th row/column

    run = {
        "engine": "augmecon-py",
        "engine_backend": backend,
        "case": case.name,
        "scenario": scenario.name,
        "workers": 1,
        "repeat": repeat,
        "runtime_seconds": runtime,
        "hv_indicator": None,
        "pareto_points": len(runner.pareto_front),
        "unique_points": len(runner.all_solutions),
        "models_solved": int(runner.models_solved),
        "models_infeasible": int(runner.infeasibilities),
    }
    return run, _signature(pareto, payoff)


_RUNNERS = {
    "pyaugmecon": _run_pyaugmecon,
    "augmecon-py": _run_augmecon_py,
}


def run_engine(
    engine: str,
    case: BenchmarkCase,
    scenario: Scenario,
    repeat: int,
    solver_name: str,
    solver_opts: dict[str, Any],
    log_dir: Path,
    workers: int,
) -> tuple[dict[str, Any], Signature]:
    """Run a benchmark via the named engine."""
    runner = _RUNNERS.get(engine)
    if runner is None:
        raise ValueError(
            f"Unknown engine {engine!r}. Available: {', '.join(_RUNNERS)}."
        )
    return runner(case, scenario, repeat, solver_name, solver_opts, log_dir, workers)

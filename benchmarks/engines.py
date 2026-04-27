"""Engine adapters for the benchmark suite.

`pyaugmecon` is this project. `augmecon-py` is the reference single-process
`MoipAugmeconR` from `../augmecon-py`, available via the `benchmarks` extra
and only meaningful for single-worker scenarios.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import multiprocessing as mp
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from pyaugmecon import PyAugmecon
from pyaugmecon.config import PyAugmeconConfig
from pyaugmecon.example_models import _load_kp_matrices, kp_model

if TYPE_CHECKING:
    from .cli import Job


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    objective_count: int
    grid_points: int
    nadir_points: list[int] | None = None
    base_case: str | None = None

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> BenchmarkCase:
        nadir = cfg.get("nadir_points")
        return cls(
            name=name,
            objective_count=int(cfg["objective_count"]),
            grid_points=int(cfg["grid_points"]),
            nadir_points=None if nadir is None else [int(v) for v in nadir],
            base_case=cfg.get("base_case"),
        )

    @property
    def dataset(self) -> str:
        return self.base_case or self.name

    def build_model(self):
        return kp_model(self.dataset, self.objective_count)

    def case_opts(self) -> dict[str, object]:
        opts: dict[str, object] = {"sample_points": self.grid_points}
        if self.nadir_points is not None:
            opts["nadir_points"] = list(self.nadir_points)
        return opts


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    opts: dict[str, object] = field(default_factory=dict)


class Signature(NamedTuple):
    """Hashable, cross-engine-comparable rounded Pareto + payoff snapshot."""

    pareto: tuple[tuple[float, ...], ...]
    payoff: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {
            "pareto": [list(row) for row in self.pareto],
            "payoff": [list(row) for row in self.payoff],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Signature | None:
        if data is None:
            return None
        return cls(
            pareto=tuple(tuple(float(v) for v in row) for row in data["pareto"]),
            payoff=tuple(tuple(float(v) for v in row) for row in data["payoff"]),
        )

    @classmethod
    def from_results(
        cls,
        points: Sequence[Sequence[float]],
        payoff: np.ndarray,
        decimals: int = 9,
    ) -> Signature:
        pts = sorted({tuple(round(float(v), decimals) for v in p) for p in points})
        pay = np.round(np.asarray(payoff, dtype=float), decimals).tolist()
        return cls(
            pareto=tuple(pts),
            payoff=tuple(tuple(float(v) for v in row) for row in pay),
        )


@dataclass(slots=True)
class RunResult:
    """Structured result from a single benchmark run."""

    engine: str
    case: str
    scenario: str
    workers: int
    sample: int
    runtime_seconds: float
    pareto_points: int
    dominated_points: int
    models_solved: int
    models_infeasible: int
    grid_points: int
    objective_count: int
    solver: str
    solver_options: dict[str, Any]
    hv_indicator: float | None = None
    engine_backend: str | None = None
    unique_points: int | None = None
    signature: Signature | None = None
    matches_baseline: bool | None = None
    cross_engine_parity: bool | str | None = None
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self, dict_factory=dict)
        if self.signature is not None:
            d["_signature"] = self.signature.to_dict()
        d.pop("signature", None)
        d.pop("matches_baseline", None)
        d.pop("cross_engine_parity", None)
        if self.engine_backend is None:
            d.pop("engine_backend", None)
        if self.unique_points is None:
            d.pop("unique_points", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        return cls(
            engine=data["engine"],
            case=data["case"],
            scenario=data["scenario"],
            workers=int(data["workers"]),
            sample=int(data["sample"]),
            runtime_seconds=float(data["runtime_seconds"]),
            pareto_points=int(data["pareto_points"]),
            dominated_points=int(data.get("dominated_points", 0)),
            models_solved=int(data["models_solved"]),
            models_infeasible=int(data["models_infeasible"]),
            grid_points=int(data["grid_points"]),
            objective_count=int(data["objective_count"]),
            solver=data["solver"],
            solver_options=data.get("solver_options") or {},
            hv_indicator=data.get("hv_indicator"),
            engine_backend=data.get("engine_backend"),
            unique_points=data.get("unique_points"),
            signature=Signature.from_dict(data.get("_signature")),
            timed_out=data.get("timed_out", False),
        )


# Engine name -> supports_parallel.
ENGINES: dict[str, bool] = {
    "pyaugmecon": True,
    "augmecon-py": False,
}


def _run_pyaugmecon(job: Job, log_dir: Path) -> RunResult:
    case, scenario = job.case, job.scenario
    cfg_kwargs = {
        **case.case_opts(),
        **scenario.opts,
        "name": f"{case.name}_{scenario.name}_s{job.sample}",
        "mode": "sampled",
        "solver_name": job.solver,
        "solver_options": dict(job.solver_options),
        "workers": job.workers,
        "artifact_folder": str(log_dir),
        "write_csv": False,
        "progress_bar": False,
        "log_to_console": False,
    }

    runner = PyAugmecon(case.build_model(), PyAugmeconConfig(**cfg_kwargs))  # ty: ignore[invalid-argument-type]
    started = time.perf_counter()
    try:
        result = runner.solve()
        return RunResult(
            engine="pyaugmecon",
            case=case.name,
            scenario=scenario.name,
            workers=job.workers,
            sample=job.sample,
            runtime_seconds=result.runtime_seconds,
            hv_indicator=result.hypervolume(),
            pareto_points=result.count,
            dominated_points=max(0, result.total_points - result.count),
            models_solved=result.models_solved,
            models_infeasible=result.models_infeasible,
            grid_points=case.grid_points,
            objective_count=case.objective_count,
            solver=job.solver,
            solver_options=dict(job.solver_options),
            signature=Signature.from_results(result.points, result.payoff_table),
            timed_out=False,
        )
    except TimeoutError:
        runtime = round(time.perf_counter() - started, 2)
        return RunResult(
            engine="pyaugmecon",
            case=case.name,
            scenario=scenario.name,
            workers=job.workers,
            sample=job.sample,
            runtime_seconds=runtime,
            hv_indicator=None,
            pareto_points=0,
            dominated_points=0,
            models_solved=runner.model.models_solved.value() if runner.model else 0,
            models_infeasible=runner.model.infeasibilities.value()
            if runner.model
            else 0,
            grid_points=case.grid_points,
            objective_count=case.objective_count,
            solver=job.solver,
            solver_options=dict(job.solver_options),
            signature=None,
            timed_out=True,
        )


def _build_kp_model(case: BenchmarkCase):
    """Knapsack pyomo model in the shape augmecon-py's MoipAugmeconR expects."""
    import pyomo.environ as p  # noqa: PLC0415

    a, b, c = _load_kp_matrices(case.dataset)
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


def _run_augmecon_py(job: Job, _log_dir: Path) -> RunResult:
    case = job.case
    try:
        from augmecon_py import MoipAugmeconR  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "augmecon-py engine requires the `augmecon-py` package. "
            "Install with: `uv sync --extra benchmarks`."
        ) from exc

    def _worker(q: mp.Queue):
        try:
            solver, backend = _build_augmecon_py_solver(job.solver, job.solver_options)
            nadir = case.nadir_points
            fixed_nadirs = None if nadir is None else [None, None, *nadir]

            runner = MoipAugmeconR(
                _build_kp_model(case),
                solver=solver,
                model_name=f"{case.name}_s{job.sample}",
                min_to_nadir_undercut=1.0,
                fixed_nadirs=fixed_nadirs,
            )
            with contextlib.redirect_stdout(None):
                runner.execute()

            pareto = [
                [float(v) for v in s.objective_values]
                for s in runner.pareto_front.values()
            ]
            payoff = runner.payoff_table[1:, 1:]

            q.put(
                {
                    "backend": backend,
                    "pareto_points": len(runner.pareto_front),
                    "dominated_points": max(
                        0, len(runner.all_solutions) - len(runner.pareto_front)
                    ),
                    "unique_points": len(runner.all_solutions),
                    "models_solved": int(runner.models_solved),
                    "models_infeasible": int(runner.infeasibilities),
                    "pareto": pareto,
                    "payoff": payoff,
                }
            )
        except Exception as e:
            q.put(e)

    started = time.perf_counter()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(q,))
    proc.start()

    timeout = float(job.scenario.opts.get("process_timeout", 7200))
    proc.join(timeout)

    runtime = round(time.perf_counter() - started, 2)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        return RunResult(
            engine="augmecon-py",
            case=case.name,
            scenario=job.scenario.name,
            workers=1,
            sample=job.sample,
            runtime_seconds=runtime,
            hv_indicator=None,
            pareto_points=0,
            dominated_points=0,
            models_solved=0,
            models_infeasible=0,
            grid_points=case.grid_points,
            objective_count=case.objective_count,
            solver=job.solver,
            solver_options=dict(job.solver_options),
            signature=None,
            timed_out=True,
        )

    res = q.get() if not q.empty() else RuntimeError("augmecon-py worker crashed")
    if isinstance(res, Exception):
        raise res

    return RunResult(
        engine="augmecon-py",
        engine_backend=res["backend"],
        case=case.name,
        scenario=job.scenario.name,
        workers=1,
        sample=job.sample,
        runtime_seconds=runtime,
        hv_indicator=None,
        pareto_points=res["pareto_points"],
        dominated_points=res["dominated_points"],
        unique_points=res["unique_points"],
        models_solved=res["models_solved"],
        models_infeasible=res["models_infeasible"],
        grid_points=case.grid_points,
        objective_count=case.objective_count,
        solver=job.solver,
        solver_options=dict(job.solver_options),
        signature=Signature.from_results(res["pareto"], res["payoff"]),
        timed_out=False,
    )


_RUNNERS = {
    "pyaugmecon": _run_pyaugmecon,
    "augmecon-py": _run_augmecon_py,
}


def run_engine(job: Job, log_dir: Path) -> RunResult:
    """Run a benchmark via the engine named on `job`."""
    runner = _RUNNERS.get(job.engine)
    if runner is None:
        raise ValueError(
            f"Unknown engine {job.engine!r}. Available: {', '.join(_RUNNERS)}."
        )
    return runner(job, log_dir)

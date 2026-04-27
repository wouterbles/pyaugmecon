from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pyomo.environ as pyo

from pyaugmecon.config import PyAugmeconConfig

# Backend preference order: direct/appsi first, persistent last.
# AUGMECON's inner loop only mutates one Param value per solve, so the
# overhead of persistent-backend Python-side change tracking (`solver.update()`)
# tends to dominate the cost of rebuilding via direct backends. Users who
# measure otherwise can pass the persistent name explicitly via `solver_name`.
SOLVER_FALLBACKS = {
    "highs": ("appsi_highs",),
    "gurobi": ("gurobi_direct", "appsi_gurobi", "gurobi_persistent", "gurobi"),
    "cplex": ("appsi_cplex", "cplex_persistent", "cplex"),
    "xpress": ("xpress_direct", "xpress_persistent", "xpress"),
    "cbc": ("cbc", "appsi_cbc"),
    "scip": ("scip",),
}

# Some Python packages provide a solver runtime without necessarily exposing the
# command-line executable names that Pyomo's selected backend plugins expect.
# This maps the common mismatch so errors can guide users to a working backend.
_PACKAGE_HINTS: dict[str, str] = {
    "cbc": (
        "Pyomo's CBC backends expect a `cbc` executable. Prefer installing "
        "`cbcbox` (or a system CBC binary on PATH). `cylp` alone may not make "
        "Pyomo's `cbc`/`appsi_cbc` plugins available."
    ),
    "scip": (
        "Pyomo's SCIP backend expects a `scip`/`scipampl` executable. "
        "`pyscipopt` can bundle the SCIP library for Python use, but that does "
        "not automatically satisfy Pyomo's executable-based `scip` plugin."
    ),
}

# Pyomo termination conditions that indicate the solver loaded a usable
# solution. Not necessarily globally optimal, but feasible enough that we can
# read variable values from the model.
SOLUTION_BEARING_TERMINATIONS = {
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.feasible,
    pyo.TerminationCondition.locallyOptimal,
    pyo.TerminationCondition.globallyOptimal,
}


# Per-backend behavior, kept as flat lookups:
#
# * `_PERSISTENT_BACKENDS`: bound via `set_instance` and refreshed via `update()`
#   before each solve.
# * `_GUROBI_MANAGE_ENV_BACKENDS`: accept `manage_env=True` at factory construction
#   (Gurobi only). The bare "gurobi" shim only honors it under the
#   direct/python solver_io paths; that case is handled in `_solver_factory_kwargs`.
# * `_GUROBI_CLOSE_BACKENDS`: call `close()` at teardown.
# * `_GUROBI_APPSI_BACKEND`: call `release_license()` at teardown.
_PERSISTENT_BACKENDS = frozenset(
    {"gurobi_persistent", "cplex_persistent", "xpress_persistent"}
)
_GUROBI_MANAGE_ENV_BACKENDS = frozenset({"gurobi_direct", "gurobi_persistent"})
_GUROBI_CLOSE_BACKENDS = frozenset({"gurobi", "gurobi_direct", "gurobi_persistent"})
_GUROBI_APPSI_BACKEND = "appsi_gurobi"


type PyomoSolveResultsLike = Any
type SolverLike = Any


class SolveOutcome(StrEnum):
    """Stable, backend-independent solve outcomes used throughout PyAugmecon.

    `OPTIMAL` covers globally and locally optimal terminations alike.
    `FEASIBLE_NONOPTIMAL` is rare but possible (e.g., MIP with relaxed gap).
    `INFEASIBLE_OR_UNBOUNDED` is reported by some presolvers when the
    distinction can't be made cheaply; AUGMECON treats it as infeasible.
    `LIMIT` covers any "ran out of budget" termination (time, iterations,
    user interrupt). `ERROR` is the catch-all for anything we don't recognize.
    """

    OPTIMAL = "optimal"
    FEASIBLE_NONOPTIMAL = "feasible_nonoptimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    LIMIT = "limit"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SolverSelection:
    requested: str
    resolved_backend: str
    attempts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SolveResult:
    outcome: SolveOutcome
    pyomo_status: pyo.SolverStatus
    pyomo_termination: pyo.TerminationCondition
    has_solution: bool
    backend: str


def _apply_solver_options(
    solver: SolverLike, backend_name: str, solver_opts: Mapping[str, object]
) -> None:
    """Push user-supplied solver options onto the backend's options container.

    `appsi_highs` uses `highs_options`; everything else uses `options`. Every
    supported backend exposes a dict-like with `update`.
    """
    if not solver_opts:
        return

    attr = "highs_options" if backend_name == "appsi_highs" else "options"
    getattr(solver, attr).update(solver_opts)


def _solver_factory_kwargs(
    config: PyAugmeconConfig, candidate: str
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if config.solver_io:
        kwargs["solver_io"] = config.solver_io

    # Why this exists (short version):
    # https://support.gurobi.com/hc/en-us/articles/38865435525777-How-do-I-use-Gurobi-with-Pyomo
    # With manage_env=True, close() releases the per-solver env/license.
    # We avoid close_global() because it affects process-global shared state.
    # See also: https://github.com/wouterbles/pyaugmecon/issues/8
    # and https://github.com/wouterbles/pyaugmecon/pull/9
    #
    # The "gurobi" shim only treats `manage_env` as meaningful under the
    # direct/python solver_io paths; other I/Os go through a different shim.
    if candidate == "gurobi":
        manage_env = config.solver_io in {"direct", "python"}
    else:
        manage_env = candidate in _GUROBI_MANAGE_ENV_BACKENDS
    if manage_env:
        kwargs["manage_env"] = True

    return kwargs


def release_solver(solver: SolverLike, backend_name: str) -> None:
    """Release backend resources for Gurobi backends.

    For direct/persistent Gurobi interfaces we set ``manage_env=True`` during
    construction, so ``close()`` releases both the model and its dedicated
    environment. APPSI Gurobi exposes ``release_license()``.
    Ref: https://support.gurobi.com/hc/en-us/articles/38865435525777-How-do-I-use-Gurobi-with-Pyomo
    """
    if backend_name in _GUROBI_CLOSE_BACKENDS:
        method = getattr(solver, "close", None)
    elif backend_name == _GUROBI_APPSI_BACKEND:
        method = getattr(solver, "release_license", None)
    else:
        return

    if callable(method):
        method()


# Errors that signal a backend is genuinely unusable (missing dependency, bad
# license, broken install, malformed config). We catch only these so that
# programmer bugs and unexpected exceptions still surface immediately.
_SOLVER_CANDIDATE_ERRORS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    ImportError,
    OSError,
)


def select_solver(
    config: PyAugmeconConfig,
) -> tuple[SolverLike, SolverSelection]:
    """
    Resolve the requested solver family to the first usable backend.

    Families such as `highs` and `gurobi` map to multiple Pyomo plugins. This
    function tries the configured fallbacks in priority order and returns the
    first backend that can be constructed and marked available.
    """
    requested = str(config.solver_name)
    candidates = SOLVER_FALLBACKS.get(requested, (requested,))
    attempts: list[str] = []
    failures: list[str] = []

    for candidate in candidates:
        attempts.append(candidate)
        kwargs = _solver_factory_kwargs(config, candidate)
        solver = None
        try:
            solver = pyo.SolverFactory(candidate, **kwargs)
            if solver is None:
                raise RuntimeError("factory returned None")

            available = getattr(solver, "available", None)
            if callable(available) and not available(exception_flag=False):
                raise RuntimeError("backend is unavailable in this environment")

            _apply_solver_options(solver, candidate, config.solver_options)
            selection = SolverSelection(
                requested=requested,
                resolved_backend=candidate,
                attempts=tuple(attempts),
            )
            return solver, selection
        except _SOLVER_CANDIDATE_ERRORS as exc:
            if solver is not None:
                with contextlib.suppress(Exception):
                    release_solver(solver, candidate)
            failures.append(f"{candidate}: {exc}")

    detail = "; ".join(failures) if failures else "no candidate backends attempted"
    package_hint = ""
    hint = _PACKAGE_HINTS.get(requested)
    if hint:
        package_hint = f" {hint}"
    raise RuntimeError(
        "Could not resolve requested solver "
        f"'{requested}'. Tried {', '.join(candidates)}. Details: {detail}. "
        f"Install/configure one of the backends and verify license availability.{package_hint}"
    )


def _prepare_persistent_backend(
    model: pyo.ConcreteModel,
    solver: SolverLike,
    backend_name: str,
) -> None:
    """For persistent backends, bind the model via set_instance/update.

    Every Pyomo persistent backend (gurobi/cplex/xpress) exposes both methods,
    so we call them directly. AUGMECON solves the same model thousands of times,
    so the `_attached_model_id` cache avoids re-binding on every solve.
    """
    if backend_name not in _PERSISTENT_BACKENDS:
        return

    model_id = id(model)
    if getattr(solver, "_pyaugmecon_attached_model_id", None) != model_id:
        solver.set_instance(model)
        solver._pyaugmecon_attached_model_id = model_id

    solver.update()


# Mapping from raw Pyomo termination conditions to our stable `SolveOutcome`.
# Anything not listed here falls through to `SolveOutcome.ERROR` so that
# unexpected solver states are loud rather than silently mis-categorized.
_TERMINATION_TO_OUTCOME: dict[pyo.TerminationCondition, SolveOutcome] = {
    pyo.TerminationCondition.optimal: SolveOutcome.OPTIMAL,
    pyo.TerminationCondition.globallyOptimal: SolveOutcome.OPTIMAL,
    pyo.TerminationCondition.locallyOptimal: SolveOutcome.OPTIMAL,
    pyo.TerminationCondition.feasible: SolveOutcome.FEASIBLE_NONOPTIMAL,
    pyo.TerminationCondition.infeasible: SolveOutcome.INFEASIBLE,
    pyo.TerminationCondition.unbounded: SolveOutcome.UNBOUNDED,
    pyo.TerminationCondition.infeasibleOrUnbounded: SolveOutcome.INFEASIBLE_OR_UNBOUNDED,
    pyo.TerminationCondition.maxTimeLimit: SolveOutcome.LIMIT,
    pyo.TerminationCondition.maxIterations: SolveOutcome.LIMIT,
    pyo.TerminationCondition.userInterrupt: SolveOutcome.LIMIT,
}


def normalize_outcome(
    _status: pyo.SolverStatus,
    termination: pyo.TerminationCondition,
) -> SolveOutcome:
    """Map a Pyomo termination condition to a stable outcome enum.

    `_status` is intentionally unused: across all supported backends the
    termination condition carries the dispositive information, and several
    backends report `status=warning` together with `termination=optimal` for
    perfectly valid solves. The two-argument signature is kept for the public
    test surface.
    """
    return _TERMINATION_TO_OUTCOME.get(termination, SolveOutcome.ERROR)


def solve_once(
    model: pyo.ConcreteModel,
    solver: SolverLike,
    backend_name: str,
    warmstart: bool,
) -> tuple[PyomoSolveResultsLike, SolveResult]:
    """
    Solve once with consistent kwarg handling and result normalization.

    APPSI and persistent backends materialize the solution via
    `solver.load_vars()`; classic backends rely on
    `model.solutions.load_from(result)`. Pyomo's `scip` plugin rejects the
    `warmstart` kwarg, so we only pass it to backends that accept it.
    """
    _prepare_persistent_backend(model, solver, backend_name)
    solve_kwargs: dict[str, object] = {"load_solutions": False}
    if backend_name != "scip":
        solve_kwargs["warmstart"] = warmstart

    result = solver.solve(model, **solve_kwargs)

    status = result.solver.status
    term = result.solver.termination_condition
    has_solution = term in SOLUTION_BEARING_TERMINATIONS

    if has_solution:
        load_vars = getattr(solver, "load_vars", None)
        if callable(load_vars):
            load_vars()
        else:
            model.solutions.load_from(result)

    return result, SolveResult(
        outcome=normalize_outcome(status, term),
        pyomo_status=status,
        pyomo_termination=term,
        has_solution=has_solution,
        backend=backend_name,
    )

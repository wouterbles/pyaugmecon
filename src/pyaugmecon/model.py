from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import prod
from typing import Any

import numpy as np
import pyomo.environ as pyo
from loguru import logger as log
from pyomo.core.base import (
    ConstraintList,
    NonNegativeReals,
    Param,
    Set,
    Var,
    maximize,
    minimize,
)

from pyaugmecon.config import PyAugmeconConfig
from pyaugmecon.helper import Counter, ProgressBar
from pyaugmecon.solver_adapter import (
    SolveOutcome,
    release_solver,
    select_solver,
    solve_once,
)

# Component names PyAugmecon attaches to the user's Pyomo model. Reserved so
# `check_user_model` can reject conflicts up front rather than overwriting
# user-defined data.
AUGMECON_RESERVED_NAMES = frozenset(
    {
        "_pyaugmecon_os",
        "_pyaugmecon_slack",
        "_pyaugmecon_eps",
        "_pyaugmecon_constraint_list",
        "_pyaugmecon_payoff_constraint_list",
    }
)


def check_user_model(user_model: pyo.ConcreteModel) -> None:
    """Reject models that already use PyAugmecon's reserved component names."""
    names_in_user_model = {
        component.name for component in user_model.component_objects()
    }
    conflicts = names_in_user_model & AUGMECON_RESERVED_NAMES
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} of your pyomo model names raised a conflict"
            f" with PyAugmecon reserved component names.\n"
            "To avoid errors, you must rename the following components: "
            f"{conflicts}"
        )


# Outcomes treated as "no feasible solution". `INFEASIBLE_OR_UNBOUNDED` is
# bundled with `INFEASIBLE` because some solvers report it when presolve
# can't disambiguate; AUGMECON's primary objective is bounded by construction
# (payoff table), so the ambiguous outcome means infeasibility in practice.
_INFEASIBLE_OUTCOMES = frozenset(
    {SolveOutcome.INFEASIBLE, SolveOutcome.INFEASIBLE_OR_UNBOUNDED}
)


class Model:
    def __init__(self, model: pyo.ConcreteModel, config: PyAugmeconConfig):
        self.model = model
        self.config = config

        # Convention: objective 0 is the primary (augmented) objective;
        # objectives 1..n-1 become the epsilon-constrained ones.
        self.n_obj = len(model.obj_list)  # ty: ignore[invalid-argument-type]
        self.iter_obj = range(self.n_obj)
        self.constrained_objectives = list(range(1, self.n_obj))

        # Shared-memory counters that workers update during a run.
        self.models_solved = Counter()
        self.infeasibilities = Counter()

        # Per-objective grid attributes; populated by `find_obj_range`.
        self.nadir_by_obj: dict[int, float] = {}
        self.obj_max_by_obj: dict[int, float] = {}
        self.obj_range_by_obj: dict[int, float] = {}
        self.level_step_by_obj: dict[int, float] = {}
        self.epsilon_levels_by_obj: dict[int, np.ndarray] = {}
        self.outcome: SolveOutcome | None = None
        self.constrained_order_outer: list[int] = list(self.constrained_objectives)
        self.constrained_order_inner: list[int] = list(
            reversed(self.constrained_order_outer)
        )
        self.grid_sizes_inner: list[int] = []
        self.grid_point_count = 0
        # `n_obj**2` accounts for the lexicographic payoff matrix construction.
        self.setup_solve_count = self.n_obj**2

        # Best-effort progress-bar total. Refined in `find_obj_range` once the
        # actual grid sizes are known. The nadir term only contributes when
        # nadir points must be auto-computed.
        constrained_count = len(self.constrained_objectives)
        nadir_solves = 0 if config.nadir_points is not None else constrained_count
        points = config.get_points_per_objective(constrained_count)
        self.to_solve = (
            self.setup_solve_count + nadir_solves + (int(prod(points)) if points else 0)
        )
        self.progress = ProgressBar(
            Counter(),
            self.to_solve,
            enabled=config.progress_bar,
        )
        self.progress.set_message("Setup")

    def obj(self, i: int) -> Any:
        """Return the i-th objective (translates 0-based to Pyomo's 1-based index)."""
        return self.model.obj_list[i + 1]  # ty: ignore[not-subscriptable]

    def deactivate_all_objectives(self) -> None:
        for i in self.iter_obj:
            self.obj(i).deactivate()

    def _make_reusable_solver(self) -> tuple[Any, str] | None:
        """Create a solver that can be reused across setup solves.

        APPSI and persistent backends can't be safely reused across
        objective activations during setup, so return None to fall
        back to one-shot solver creation per solve call.
        """
        solver, selection = select_solver(self.config)
        backend = selection.resolved_backend
        if backend.startswith("appsi_") or backend.endswith("_persistent"):
            release_solver(solver, backend)
            return None
        return (solver, backend)

    @contextmanager
    def _reusable_solver(self) -> Iterator[tuple[Any, str] | None]:
        """Yield a reusable solver (or None) and release it on exit."""
        reusable = self._make_reusable_solver()
        try:
            yield reusable
        finally:
            if reusable is not None:
                release_solver(reusable[0], reusable[1])

    def solve(self) -> None:
        """Solve the model with a freshly-created solver and release it."""
        solver, selection = select_solver(self.config)
        try:
            self._solve_with(solver, selection.resolved_backend)
        finally:
            release_solver(solver, selection.resolved_backend)

    def _solve_with(self, solver: Any, backend: str) -> None:
        """Solve using a borrowed solver (caller owns its lifecycle)."""
        _, solve_result = solve_once(
            self.model, solver, backend, warmstart=self.config.solve_warmstart
        )
        self.outcome = solve_result.outcome

    def is_infeasible(self) -> bool:
        return self.outcome in _INFEASIBLE_OUTCOMES

    def min_to_max(self) -> None:
        """Convert minimize objectives to maximize form in-place.

        The epsilon-constraint loop compares objective values assuming every
        objective is maximized. We track the original sense in `obj_goal` so
        `pyaugmecon.py` can flip final results back to the user's orientation.
        """
        self.obj_goal = [
            -1 if self.obj(o).sense == minimize else 1 for o in self.iter_obj
        ]
        for o in self.iter_obj:
            objective = self.obj(o)
            if objective.sense == minimize:
                objective.sense = maximize
                objective.expr = -objective.expr

    def _solve_objective(
        self, objective_idx: int, reusable: tuple[Any, str] | None, context: str
    ) -> float:
        """Activate `objective_idx`, solve, and return the optimal value.

        The try/finally guarantees the model is left with all objectives
        deactivated so the next caller can activate exactly one.
        """
        objective = self.obj(objective_idx)
        objective.activate()
        try:
            if reusable is None:
                self.solve()
            else:
                self._solve_with(*reusable)
            self.progress.increment()
            if self.outcome != SolveOutcome.OPTIMAL:
                raise RuntimeError(f"{context} failed (outcome={self.outcome}).")
            value = objective()
            if value is None:
                # Pyomo returns None when no incumbent is loaded; treat as a
                # solver-side bug rather than propagating a NaN.
                raise RuntimeError(f"{context} returned no value.")
            return float(value)
        finally:
            objective.deactivate()

    def construct_payoff(self) -> None:
        """Build the lexicographic payoff matrix.

        For each row `i`:
          1. Solve max f_i(x) on its own and record `payoff[i, i]`.
          2. Lock `f_i(x) == payoff[i, i]` and lex-optimize each `f_j` (j != i),
             recording `payoff[i, j]` and locking each result in turn.
          3. Clear the transient constraint list before moving to row `i+1`.

        The diagonal pass (step 1) is split out so all diagonal entries are
        filled before any equality constraint is added.
        """
        self.progress.set_message("Setup")

        with self._reusable_solver() as reusable:
            self.payoff = np.full((self.n_obj, self.n_obj), np.inf)
            payoff_cl = ConstraintList()
            self.model._pyaugmecon_payoff_constraint_list = payoff_cl

            # Pass 1: diagonal.
            for i in self.iter_obj:
                self.payoff[i, i] = self._solve_objective(
                    i, reusable, f"Payoff construction for objective pair ({i}, {i})"
                )

            # Pass 2: per row `i`, lock f_i and lex-optimize the others.
            for i in self.iter_obj:
                payoff_cl.add(expr=self.obj(i).expr == self.payoff[i, i])  # ty: ignore[missing-argument]

                for j in self.iter_obj:
                    if i == j:
                        continue
                    self.payoff[i, j] = self._solve_objective(
                        j,
                        reusable,
                        f"Payoff construction for objective pair ({i}, {j})",
                    )
                    payoff_cl.add(expr=self.obj(j).expr == self.payoff[i, j])  # ty: ignore[missing-argument]

                # Reset before the next row; this row's equalities would
                # over-constrain the next.
                payoff_cl.clear()

    def _compute_auto_safe_nadirs(self) -> dict[int, float]:
        """Compute "safe" nadir lower bounds by minimizing each constrained objective.

        After `min_to_max` every objective is in maximization form, so the true
        nadir of `f_j` over the Pareto front is bounded below by the
        unconstrained min of `f_j`. We flip each constrained objective's sense
        to `minimize` to get that bound, then restore it.

        These bounds never cut off Pareto-optimal points; looser bounds only
        cost extra (infeasible) grid solves, never correctness.
        """
        log.debug("Computing auto-safe nadir bounds")
        bounds: dict[int, float] = {}

        with self._reusable_solver() as reusable:
            self.deactivate_all_objectives()
            for objective_idx in self.constrained_objectives:
                objective = self.obj(objective_idx)
                original_sense = objective.sense
                objective.sense = minimize
                try:
                    bound = self._solve_objective(
                        objective_idx,
                        reusable,
                        f"Auto-safe nadir bound for objective {objective_idx + 1}",
                    )
                finally:
                    # Restore maximization so later passes see the original sense.
                    objective.sense = original_sense

                if not np.isfinite(bound):
                    raise RuntimeError(
                        f"Auto-safe nadir for objective {objective_idx + 1} is non-finite: {bound}."
                    )
                bounds[objective_idx] = bound

        return bounds

    def find_obj_range(self) -> None:
        """Build the per-objective epsilon grids and finalize traversal order.

        For each constrained objective we resolve a lower bound (nadir) and an
        upper bound (max from the payoff table) and discretize the interval
        into either an integer-stepped exact grid or a uniformly sampled grid.
        The objectives are then ordered: in `auto_range` mode the widest range
        changes slowest in the grid. This gives AUGMECON-R more neighboring
        points where an infeasible solve can skip later solves.
        """
        self.nadir_by_obj = self._resolve_nadirs()
        self.obj_max_by_obj = {}
        self.obj_range_by_obj = {}
        self.level_step_by_obj = {}
        self.epsilon_levels_by_obj = {}

        constrained_count = len(self.constrained_objectives)
        points_per_obj = self.config.get_points_per_objective(constrained_count)
        is_exact = self.config.mode == "exact"

        for pos, objective_idx in enumerate(self.constrained_objectives):
            n_points = points_per_obj[pos] if not is_exact else None
            self._build_and_record_levels(objective_idx, n_points, is_exact)

        if self.config.objective_order == "auto_range":
            # Widest range first; ties broken by index for determinism.
            self.constrained_order_outer = sorted(
                self.constrained_objectives,
                key=lambda idx: (-self.obj_range_by_obj[idx], idx),
            )
        else:
            self.constrained_order_outer = list(self.constrained_objectives)

        # Inner loop iterates fastest, so reverse the outer order to keep the
        # widest dimension as the innermost axis of the 1D linearization that
        # workers decode via divmod (see `solver_process.py`).
        self.constrained_order_inner = list(reversed(self.constrained_order_outer))
        self.grid_sizes_inner = [
            int(self.epsilon_levels_by_obj[obj_idx].size)
            for obj_idx in self.constrained_order_inner
        ]
        self.grid_point_count = int(prod(self.grid_sizes_inner))

        # Refine the progress-bar total now that we know the actual grid.
        self.setup_solve_count = self.n_obj**2 + (
            0 if self.config.nadir_points is not None else constrained_count
        )
        self.progress.set_total(self.setup_solve_count + self.grid_point_count)

        self.epsilon_grid = np.array(
            [
                self.epsilon_levels_by_obj[obj_idx]
                for obj_idx in self.constrained_order_inner
            ],
            dtype=object,
        )

    def _resolve_nadirs(self) -> dict[int, float]:
        """Pick explicit nadir points from config or compute auto-safe ones.

        User-provided nadirs are given in the original min/max orientation, so
        we multiply by `obj_goal` to re-orient them into the maximization frame
        used internally.
        """
        if self.config.nadir_points is None:
            return self._compute_auto_safe_nadirs()
        return {
            objective_idx: float(self.config.nadir_points[pos])
            * self.obj_goal[objective_idx]
            for pos, objective_idx in enumerate(self.constrained_objectives)
        }

    def _build_and_record_levels(
        self, objective_idx: int, n_points: int | None, is_exact: bool
    ) -> None:
        """Build the epsilon-level array for one objective and cache its
        derived stats (min/max/range/step).

        Exact mode walks an integer grid (step 1) and rejects fractional
        bounds. Sampled mode uses a uniform `linspace` of `n_points` levels.
        Recorded min/max are taken from the array so any rounding (e.g. exact
        snapping) is reflected consistently downstream.
        """
        obj_min = self.nadir_by_obj[objective_idx]
        obj_max = float(np.max(self.payoff[:, objective_idx]))

        if not (np.isfinite(obj_min) and np.isfinite(obj_max)):
            raise RuntimeError(
                f"Objective {objective_idx + 1} has non-finite bounds: "
                f"min={obj_min}, max={obj_max}."
            )

        if is_exact:
            low, high = round(obj_min), round(obj_max)
            # Tolerate small solver float noise; reject genuinely fractional bounds.
            if abs(obj_min - low) > 1e-6 or abs(obj_max - high) > 1e-6:
                raise ValueError(
                    f"Exact mode requires integer grid bounds for objective "
                    f"{objective_idx + 1}: min={obj_min}, max={obj_max}."
                )
            if low > high:
                raise RuntimeError(
                    f"Invalid exact grid for objective {objective_idx + 1}: "
                    f"min={low} > max={high}."
                )
            levels = np.arange(low, high + 1, dtype=float)
        else:
            assert n_points is not None
            if n_points < 2:
                raise ValueError(
                    f"Sampled mode requires >= 2 points per objective; "
                    f"objective {objective_idx + 1} got {n_points}."
                )
            if obj_min > obj_max:
                raise RuntimeError(
                    f"Invalid sampled grid for objective {objective_idx + 1}: "
                    f"min={obj_min} > max={obj_max}."
                )
            levels = np.linspace(obj_min, obj_max, num=n_points, dtype=float)

        if levels.size == 0:
            raise RuntimeError(
                f"Objective {objective_idx + 1} produced an empty epsilon grid."
            )

        self.obj_max_by_obj[objective_idx] = float(levels[-1])
        self.nadir_by_obj[objective_idx] = float(levels[0])
        raw_range = float(levels[-1] - levels[0])
        # Avoid a zero range (would divide by zero in the augmented penalty).
        # A degenerate single-level grid behaves as if the range were unit-scale.
        self.obj_range_by_obj[objective_idx] = raw_range if raw_range > 0 else 1.0
        self.level_step_by_obj[objective_idx] = (
            float(levels[1] - levels[0]) if levels.size > 1 else 1.0
        )
        self.epsilon_levels_by_obj[objective_idx] = levels

    def convert_prob(self) -> None:
        """Convert the multi-objective model to augmented epsilon-constraint form.

        For each constrained objective i (i = 1..n-1) attach:

            f_i(x) - slack_i == eps_i             (epsilon constraint)

        and augment the primary objective f_0 with a lex-prioritized slack
        penalty:

            max  f_0(x) + delta * sum_i ( 10^(-pos_i) / range_i ) * slack_i

        where `delta = config.penalty_weight`, `10^(-pos_i)` enforces lex
        priority across objectives (first inner gets weight 1, next 1e-1,
        etc.), and `range_i` normalizes by each objective's payoff range.

        Slack is non-negative, so the equality gives f_i(x) >= eps_i.
        Maximizing the augmented objective drives slack toward zero, recovering
        epsilon-constraint behavior while remaining strictly Pareto-efficient.
        """
        constraint_list = ConstraintList()
        self.model._pyaugmecon_constraint_list = constraint_list

        # Pyomo components are 1-based; convert from our 0-based objective indices.
        objective_component_indices = [
            obj_idx + 1 for obj_idx in self.constrained_order_inner
        ]
        objective_set = Set(ordered=True, initialize=objective_component_indices)
        self.model._pyaugmecon_os = objective_set
        slack_var = Var(objective_set, within=NonNegativeReals)
        self.model._pyaugmecon_slack = slack_var
        # Mutable Param so workers can update epsilon per grid point without
        # rebuilding constraints.
        epsilon_param = Param(objective_set, within=pyo.Any, mutable=True)
        self.model._pyaugmecon_eps = epsilon_param

        primary_objective = self.obj(0)
        for pos, objective_idx in enumerate(self.constrained_order_inner):
            # 10^(-pos) gives strict lexicographic priority.
            weight = 10 ** (-pos)
            component_idx = objective_idx + 1
            primary_objective.expr += self.config.penalty_weight * (
                weight * slack_var[component_idx] / self.obj_range_by_obj[objective_idx]
            )

            constraint_list.add(  # ty: ignore[missing-argument]
                expr=self.obj(objective_idx).expr - slack_var[component_idx]
                == epsilon_param[component_idx]
            )

"""Solve results and solution records."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymoo.indicators.hv import HV

type Point = tuple[float, ...]
type VariableValues = dict[object, float]
type DecisionVariables = dict[str, VariableValues]


@dataclass(frozen=True, slots=True)
class Solution:
    """One Pareto point with optional decision-variable values."""

    point: Point
    variables: DecisionVariables | None = None


type WorkerChunk = list[Solution]


@dataclass(slots=True)
class PyAugmeconResult:
    """Final solve result."""

    solutions: tuple[Solution, ...]
    payoff_table: np.ndarray
    runtime_seconds: float
    models_solved: int
    models_infeasible: int
    visited_points: int
    grid_point_count: int
    decision_variables_stored: bool
    total_points: int
    _hypervolume: float | None = field(default=None, init=False, repr=False)

    @staticmethod
    def undominated_mask(points: np.ndarray) -> np.ndarray:
        """Boolean mask of non-dominated rows (maximization)."""
        keep = np.ones(points.shape[0], dtype=bool)
        for i, point in enumerate(points):
            if keep[i]:
                keep[keep] = np.any(points[keep] > point, axis=1)
                keep[i] = True
        return keep

    @property
    def points(self) -> tuple[Point, ...]:
        return tuple(s.point for s in self.solutions)

    @property
    def count(self) -> int:
        return len(self.solutions)

    @property
    def skipped_points(self) -> int:
        return max(0, self.visited_points - self.models_solved)

    def hypervolume(self) -> float:
        """Hypervolume of the Pareto front. Cached after first call."""
        if self._hypervolume is None:
            if not self.solutions:
                self._hypervolume = 0.0
            else:
                indicator = HV(ref_point=np.diag(self.payoff_table))
                self._hypervolume = float(indicator(np.array(self.points, dtype=float)))
        return self._hypervolume

    def solution_for(self, point: Point) -> Solution:
        """Look up a solution by its point (as seen in `points`)."""
        key = tuple(float(v) for v in point)
        for solution in self.solutions:
            if solution.point == key:
                return solution
        raise ValueError(f"Solution not found: {point}")

    def variables_for(self, point: Point) -> DecisionVariables:
        """Decision variables for a Pareto point.

        Requires `store_decision_variables=True` on the config.
        """
        if not self.decision_variables_stored:
            raise RuntimeError(
                "Decision-variable extraction is disabled. "
                "Set `store_decision_variables=True` to enable it."
            )
        solution = self.solution_for(point)
        if solution.variables is None:
            raise RuntimeError("No decision variables are stored for this solution.")
        return solution.variables

    @classmethod
    def from_worker_chunks(
        cls,
        worker_chunks: list[WorkerChunk],
        *,
        sign: tuple[int, ...],
        payoff_table: np.ndarray,
        runtime_seconds: float,
        models_solved: int,
        models_infeasible: int,
        visited_points: int,
        grid_point_count: int,
        decision_variables_stored: bool,
        round_decimals: int,
    ) -> PyAugmeconResult:
        """Build a result from worker output.

        Round to dedupe near-duplicates, drop dominated points, then flip
        signs back to the user's original min/max frame.
        """
        # Round each point and keep one solution per rounded key (first wins).
        unique: dict[Point, Solution] = {}
        for chunk in worker_chunks:
            for sol in chunk:
                key = tuple(round(float(v), round_decimals) for v in sol.point)
                unique.setdefault(key, Solution(key, sol.variables))

        # Drop dominated solutions, then flip signs back to the user's
        # original min/max frame and sort.
        sign_arr = np.array(sign, dtype=float)
        if unique:
            kept_mask = PyAugmeconResult.undominated_mask(
                np.array(list(unique), dtype=float)
            )
            solutions = tuple(
                sorted(
                    (
                        Solution(tuple(np.array(s.point) * sign_arr), s.variables)
                        for s, keep in zip(unique.values(), kept_mask, strict=False)
                        if keep
                    ),
                    key=lambda s: s.point,
                )
            )
        else:
            solutions = ()

        payoff = np.asarray(payoff_table, dtype=float) * sign_arr
        payoff.flags.writeable = False

        return cls(
            solutions=solutions,
            payoff_table=payoff,
            runtime_seconds=runtime_seconds,
            models_solved=models_solved,
            models_infeasible=models_infeasible,
            visited_points=visited_points,
            grid_point_count=grid_point_count,
            decision_variables_stored=decision_variables_stored,
            total_points=len(unique),
        )

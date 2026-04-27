"""
Bundled knapsack models used by tests, examples, and benchmarks.

These helpers load CSV-backed knapsack instances shipped inside the package and
expose them as ready-to-solve Pyomo models with `obj_list`.
"""

from __future__ import annotations

from importlib import resources
from typing import IO, Any

import numpy as np
from pyomo.core.base import (
    Binary,
    ConcreteModel,
    Constraint,
    ObjectiveList,
    Set,
    Var,
    maximize,
)


def _read_indexed_csv(source: str | IO[str]) -> np.ndarray:
    """Load a numeric CSV with header row and index column; return the data body."""
    data = np.genfromtxt(source, delimiter=",", skip_header=1, filling_values=0.0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 1:]


def _kp_input_dir(model_type: str) -> resources.abc.Traversable:
    """
    Resolve the CSV directory for a bundled knapsack dataset.

    Use package resources instead of filesystem paths so the examples keep
    working after installation from a wheel.
    """
    dataset_dir = resources.files("pyaugmecon").joinpath("data", model_type)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            "Knapsack input dataset was not found: "
            f"{dataset_dir}. Use one of the bundled test datasets (for example: "
            "`2kp50`, `3kp40`, `4kp40`)."
        )
    return dataset_dir


def _load_kp_matrices(model_type: str) -> tuple[Any, Any, Any]:
    """Load the constraint matrix, capacities, and objective weights."""
    dataset_dir = _kp_input_dir(model_type)

    def _read(name: str) -> Any:
        with dataset_dir.joinpath(name).open("r", encoding="utf-8") as handle:
            return _read_indexed_csv(handle)

    return _read("a.csv"), _read("b.csv"), _read("c.csv")


def kp_model(model_type: str, objective_count: int) -> ConcreteModel:
    """
    Build a multi-objective knapsack model from a bundled CSV dataset.

    The resulting model follows the interface expected by `PyAugmecon`:
    objective functions live in `obj_list` and are deactivated before solve.
    """
    a, b, c = _load_kp_matrices(model_type)

    if any(len(matrix) < objective_count for matrix in (a, b, c)):
        raise ValueError(
            f"Dataset `{model_type}` does not contain enough rows for {objective_count} "
            "objectives/constraints."
        )

    model = ConcreteModel()
    model.ITEMS = Set(initialize=range(len(a[0])))
    model.x = Var(model.ITEMS, within=Binary)
    items: Any = model.ITEMS
    x: Any = model.x

    model.ROW = Set(initialize=range(objective_count))

    def constraint_rule(_model: ConcreteModel, row: int):
        return sum(a[row][i] * x[i] for i in items) <= b[row][0]

    model.cons = Constraint(model.ROW, rule=constraint_rule)

    model.obj_list = ObjectiveList()
    for row in range(objective_count):
        model.obj_list.add(
            expr=sum(c[row][i] * x[i] for i in items),
            sense=maximize,
        )

    return model

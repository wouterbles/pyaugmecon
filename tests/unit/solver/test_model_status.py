import pyomo.environ as pyo
import pytest

from pyaugmecon.solver.model import Model
from tests.support.factories import make_config


def _solver_available(solver_name: str) -> bool:
    solver = pyo.SolverFactory(solver_name)
    if solver is None:
        return False
    if not hasattr(solver, "available"):
        return True
    return bool(solver.available(exception_flag=False))


@pytest.mark.parametrize("solver_name", ["appsi_highs", "highs"])
def test_infeasible_status_does_not_raise_when_no_solution_loaded(
    solver_name: str,
    infeasible_model,
):
    if not _solver_available(solver_name):
        pytest.skip(f"solver {solver_name!r} not available")

    config = make_config(
        f"infeasible_{solver_name}",
        solver_name=solver_name,
    )

    model = Model(infeasible_model, config)
    model.obj(0).activate()

    model.solve()

    assert model.is_infeasible()

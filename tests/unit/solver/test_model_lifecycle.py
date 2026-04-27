import pyomo.environ as pyo
import pytest

from pyaugmecon.solver.adapter import SolverSelection
from pyaugmecon.solver.model import Model
from tests.support.factories import make_config


@pytest.fixture
def minimal_two_objective_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, None))
    model.obj_list = pyo.ObjectiveList()
    model.obj_list.add(expr=model.x, sense=pyo.maximize)
    model.obj_list.add(expr=model.x, sense=pyo.maximize)
    return model


def test_unreusable_solver_probe_is_released(mocker, minimal_two_objective_model):
    solver = object()
    released = []
    mock_select = mocker.patch(
        "pyaugmecon.solver.model.select_solver",
        return_value=(
            solver,
            SolverSelection(
                requested="gurobi",
                resolved_backend="appsi_gurobi",
                attempts=("appsi_gurobi",),
            ),
        ),
    )
    mocker.patch(
        "pyaugmecon.solver.model.release_solver",
        side_effect=lambda s, b: released.append((s, b)),
    )

    model = Model(minimal_two_objective_model, make_config("solver_lifecycle"))

    assert model._make_reusable_solver() is None
    assert released == [(solver, "appsi_gurobi")]
    mock_select.assert_called_once()

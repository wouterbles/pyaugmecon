import pyomo.environ as pyo
import pytest

from pyaugmecon.solver.adapter import SolverSelection


@pytest.fixture
def infeasible_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(within=pyo.NonNegativeReals)
    model.c = pyo.Constraint(expr=model.x <= -1)

    model.obj_list = pyo.ObjectiveList()
    model.obj_list.add(expr=model.x, sense=pyo.maximize)
    model.obj_list.add(expr=model.x, sense=pyo.maximize)

    for objective_idx in model.obj_list:
        model.obj_list[objective_idx].deactivate()

    return model


@pytest.fixture
def pyomo_result(mocker):
    result = mocker.MagicMock()
    result.solver.status = pyo.SolverStatus.ok
    result.solver.termination_condition = pyo.TerminationCondition.optimal
    return result


@pytest.fixture
def solver_selection():
    return SolverSelection(
        requested="gurobi",
        resolved_backend="appsi_gurobi",
        attempts=("appsi_gurobi",),
    )

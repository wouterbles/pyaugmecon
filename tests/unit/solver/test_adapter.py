import pyomo.environ as pyo
import pytest

from pyaugmecon.solver.adapter import (
    SolveOutcome,
    normalize_outcome,
    release_solver,
    select_solver,
    solve_once,
)
from tests.support.factories import make_config


def test_normalize_outcome_mappings():
    assert (
        normalize_outcome(pyo.SolverStatus.ok, pyo.TerminationCondition.optimal)
        == SolveOutcome.OPTIMAL
    )
    assert (
        normalize_outcome(pyo.SolverStatus.ok, pyo.TerminationCondition.feasible)
        == SolveOutcome.FEASIBLE_NONOPTIMAL
    )
    assert (
        normalize_outcome(pyo.SolverStatus.ok, pyo.TerminationCondition.infeasible)
        == SolveOutcome.INFEASIBLE
    )
    assert (
        normalize_outcome(pyo.SolverStatus.ok, pyo.TerminationCondition.unbounded)
        == SolveOutcome.UNBOUNDED
    )
    assert (
        normalize_outcome(
            pyo.SolverStatus.ok, pyo.TerminationCondition.infeasibleOrUnbounded
        )
        == SolveOutcome.INFEASIBLE_OR_UNBOUNDED
    )
    assert (
        normalize_outcome(pyo.SolverStatus.ok, pyo.TerminationCondition.maxTimeLimit)
        == SolveOutcome.LIMIT
    )
    assert (
        normalize_outcome(
            pyo.SolverStatus.error, pyo.TerminationCondition.internalSolverError
        )
        == SolveOutcome.ERROR
    )


def test_select_solver_resolves_highs_family(mocker):
    fake = mocker.MagicMock()
    fake.available.return_value = True
    mock_factory = mocker.patch(
        "pyaugmecon.solver.adapter.pyo.SolverFactory", return_value=fake
    )
    config = make_config("test_highs", solver_name="highs")

    solver, selection = select_solver(config)

    assert solver is fake
    assert selection.requested == "highs"
    assert selection.resolved_backend == "appsi_highs"
    mock_factory.assert_called_once_with("appsi_highs")


def test_select_solver_reports_missing_plugin():
    config = make_config("test_missing", solver_name="nonexistent_solver_backend")
    with pytest.raises(RuntimeError, match="Could not resolve requested solver"):
        select_solver(config)


def test_select_solver_enables_managed_env_for_gurobi_direct(mocker):
    fake = mocker.MagicMock()
    fake.available.return_value = True
    mock_factory = mocker.patch(
        "pyaugmecon.solver.adapter.pyo.SolverFactory", return_value=fake
    )
    config = make_config("test_gurobi_direct", solver_name="gurobi_direct")

    _, selection = select_solver(config)

    assert selection.resolved_backend == "gurobi_direct"
    mock_factory.assert_called_once_with("gurobi_direct", manage_env=True)


def test_select_solver_does_not_pass_managed_env_to_appsi_gurobi(mocker):
    fake = mocker.MagicMock()
    fake.available.return_value = True
    mock_factory = mocker.patch(
        "pyaugmecon.solver.adapter.pyo.SolverFactory", return_value=fake
    )
    config = make_config("test_appsi_gurobi", solver_name="appsi_gurobi")

    _, selection = select_solver(config)

    assert selection.resolved_backend == "appsi_gurobi"
    mock_factory.assert_called_once_with("appsi_gurobi")


def test_release_solver_closes_gurobi_direct_backend(mocker):
    solver = mocker.MagicMock()
    release_solver(solver, "gurobi_direct")
    solver.close.assert_called_once()
    solver.release_license.assert_not_called()


def test_release_solver_releases_appsi_gurobi_license(mocker):
    solver = mocker.MagicMock()
    release_solver(solver, "appsi_gurobi")
    solver.close.assert_not_called()
    solver.release_license.assert_called_once()


def test_release_solver_ignores_unmanaged_backend(mocker):
    solver = mocker.MagicMock()
    release_solver(solver, "glpk")
    solver.close.assert_not_called()
    solver.release_license.assert_not_called()


def test_solve_once_passes_supported_warmstart_and_load_flags(mocker, pyomo_result):
    solver = mocker.MagicMock()
    solver.solve.return_value = pyomo_result
    model = pyo.ConcreteModel()

    _, result = solve_once(model, solver, backend_name="dummy", warmstart=True)

    solver.solve.assert_called_once_with(model, load_solutions=False, warmstart=True)
    solver.load_vars.assert_called_once()
    assert result.outcome == SolveOutcome.OPTIMAL


def test_solve_once_passes_model_and_kwargs_to_vararg_solver(mocker, pyomo_result):
    solver = mocker.MagicMock()
    solver.solve.return_value = pyomo_result
    model = pyo.ConcreteModel()

    _, result = solve_once(model, solver, backend_name="dummy", warmstart=True)

    solver.solve.assert_called_once_with(model, load_solutions=False, warmstart=True)
    solver.load_vars.assert_called_once()
    assert result.outcome == SolveOutcome.OPTIMAL


def test_solve_once_prepares_persistent_backend_once_per_model(mocker, pyomo_result):
    solver = mocker.MagicMock()
    solver.solve.return_value = pyomo_result
    model = pyo.ConcreteModel()

    solve_once(model, solver, backend_name="gurobi_persistent", warmstart=True)
    solve_once(model, solver, backend_name="gurobi_persistent", warmstart=True)

    solver.set_instance.assert_called_once_with(model)
    assert solver.update.call_count == 2
    assert solver.solve.call_count == 2
    solver.solve.assert_called_with(model, load_solutions=False, warmstart=True)


def test_solve_once_rebinds_persistent_backend_on_model_change(mocker, pyomo_result):
    solver = mocker.MagicMock()
    solver.solve.return_value = pyomo_result
    model = pyo.ConcreteModel()
    model_b = pyo.ConcreteModel()

    solve_once(model, solver, backend_name="gurobi_persistent", warmstart=False)
    solve_once(model_b, solver, backend_name="gurobi_persistent", warmstart=False)

    assert solver.set_instance.call_count == 2
    solver.set_instance.assert_has_calls([mocker.call(model), mocker.call(model_b)])
    assert solver.update.call_count == 2

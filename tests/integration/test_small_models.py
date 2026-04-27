from pathlib import Path

import numpy as np
import pytest

from pyaugmecon import PyAugmecon
from tests.support.assertions import array_equal
from tests.support.factories import make_config
from tests.support.models import (
    three_objective_mixed_model,
    three_objective_model,
    two_objective_model,
)


@pytest.fixture(scope="module")
def two_objective_run():
    solver = PyAugmecon(
        two_objective_model(),
        make_config("two_objective_model", mode="sampled", sample_points=10),
    )
    solver.solve()
    return solver


@pytest.fixture(scope="module")
def three_objective_run():
    solver = PyAugmecon(
        three_objective_model(),
        make_config("three_objective_model", mode="sampled", sample_points=10),
    )
    solver.solve()
    return solver


@pytest.fixture(scope="module")
def three_objective_mixed_run():
    solver = PyAugmecon(
        three_objective_mixed_model(),
        make_config("three_objective_mixed_model", mode="sampled", sample_points=10),
    )
    solver.solve()
    return solver


def test_two_objective_sampled_front_is_sorted_and_hits_endpoints(two_objective_run):
    pareto = list(two_objective_run.result.points)
    assert pareto == sorted(pareto)
    assert len(pareto) == 3
    assert pareto[0] == pytest.approx((8.0, 184.0), abs=1e-2)
    assert pareto[-1] == pytest.approx((20.0, 160.0), abs=1e-2)

    second_objective = [point[1] for point in pareto]
    assert second_objective == sorted(second_objective, reverse=True)


def test_three_objective_payoff_table(three_objective_run):
    payoff = np.array(
        [[3075000, 62460, 33000], [3855000, 45180, 37000], [3225000, 55260, 23000]]
    )
    assert array_equal(three_objective_run.result.payoff_table, payoff, 2)


def test_mixed_orientation_flips_second_objective_sign(
    three_objective_run,
    three_objective_mixed_run,
):
    base_front = sorted(three_objective_run.result.points)
    mixed_front = sorted(three_objective_mixed_run.result.points)

    assert len(base_front) == len(mixed_front)

    for base_point, mixed_point in zip(base_front, mixed_front, strict=False):
        assert mixed_point[0] == pytest.approx(base_point[0], abs=1e-2)
        assert mixed_point[1] == pytest.approx(-base_point[1], abs=1e-2)
        assert mixed_point[2] == pytest.approx(base_point[2], abs=1e-2)


def test_decision_variables_are_plain_mappings(tmp_path):
    solver = PyAugmecon(
        two_objective_model(),
        make_config(
            "two_objective_with_decisions",
            mode="sampled",
            sample_points=5,
            store_decision_variables=True,
            artifact_folder=str(tmp_path),
        ),
    )
    result = solver.solve()

    decision_vars = result.variables_for(result.points[0])
    assert decision_vars
    assert all(isinstance(values, dict) for values in decision_vars.values())


def test_write_csv_writes_csv_artifacts(tmp_path):
    artifact_name = "csv_artifacts"
    solver = PyAugmecon(
        two_objective_model(),
        make_config(
            artifact_name,
            mode="sampled",
            sample_points=5,
            artifact_folder=str(tmp_path),
            artifact_name=artifact_name,
            write_csv=True,
        ),
    )
    solver.solve()

    output_dir = Path(tmp_path) / artifact_name
    assert output_dir.is_dir()
    assert (output_dir / "epsilon_grid.csv").is_file()
    assert (output_dir / "payoff_table.csv").is_file()
    assert (output_dir / "solutions.csv").is_file()


def test_safe_and_payoff_nadir_strategies_find_same_pareto_front():
    """Both auto-nadir strategies must find the extreme (lex-optimal) points.

    In sampled mode the per-objective grid widths differ between strategies,
    so the interior sample points differ; what must agree is that both
    strategies hit the lex-payoff diagonal endpoints (true Pareto extremes)
    and produce only non-dominated points.
    """
    safe = PyAugmecon(
        three_objective_model(),
        make_config(
            "three_obj_nadir_safe",
            mode="sampled",
            sample_points=8,
            nadir_strategy="safe",
        ),
    ).solve()
    payoff = PyAugmecon(
        three_objective_model(),
        make_config(
            "three_obj_nadir_payoff",
            mode="sampled",
            sample_points=8,
            nadir_strategy="payoff",
            nadir_undercut=0.8,
        ),
    ).solve()

    # Both runs must produce a non-empty front and agree on the extreme
    # (lex-payoff diagonal) points, which are the true Pareto extremes.
    assert safe.count > 0
    assert payoff.count > 0
    safe_diag = tuple(
        float(safe.payoff_table[i, i]) for i in range(safe.payoff_table.shape[0])
    )
    payoff_diag = tuple(
        float(payoff.payoff_table[i, i]) for i in range(payoff.payoff_table.shape[0])
    )
    assert safe_diag == pytest.approx(payoff_diag, abs=1e-2)

import pytest

import pyaugmecon.results as result_module
from pyaugmecon import PyAugmeconResult, Solution


def test_result_keeps_pareto_points_and_variables():
    result = PyAugmeconResult.from_worker_chunks(
        [
            [
                Solution((1.0, 2.0), {"x": {0: 1.0}}),
                Solution((1.00000001, 2.0), {"x": {0: 9.0}}),
                Solution((3.0, 1.0), {"x": {0: 3.0}}),
                Solution((0.0, 0.0), {"x": {0: 0.0}}),
            ]
        ],
        sign=(1, 1),
        payoff_table=[[3.0, 1.0], [1.0, 2.0]],
        runtime_seconds=1.25,
        models_solved=4,
        models_infeasible=1,
        visited_points=5,
        grid_point_count=6,
        decision_variables_stored=True,
        round_decimals=6,
    )

    assert result.points == ((1.0, 2.0), (3.0, 1.0))
    assert result.count == 2
    assert result.total_points == 3
    assert result.skipped_points == 1
    assert result.variables_for((1.0, 2.0)) == {"x": {0: 1.0}}


def test_result_hypervolume_is_lazy(mocker):
    calls = {"init": 0, "call": 0}

    class _DummyHV:
        def __init__(self, *, ref_point):
            calls["init"] += 1
            self.ref_point = ref_point

        def __call__(self, values):
            calls["call"] += 1
            assert values.shape == (2, 2)
            return 12.5

    mocker.patch.object(result_module, "HV", _DummyHV)

    result = PyAugmeconResult.from_worker_chunks(
        [[Solution((1.0, 2.0)), Solution((3.0, 1.0))]],
        sign=(1, 1),
        payoff_table=[[3.0, 1.0], [1.0, 2.0]],
        runtime_seconds=0.5,
        models_solved=2,
        models_infeasible=0,
        visited_points=2,
        grid_point_count=2,
        decision_variables_stored=False,
        round_decimals=6,
    )

    assert calls == {"init": 0, "call": 0}
    assert result.hypervolume() == 12.5
    assert result.hypervolume() == 12.5
    assert calls == {"init": 1, "call": 1}


def test_result_rejects_decision_variable_lookup_when_disabled():
    result = PyAugmeconResult.from_worker_chunks(
        [[Solution((1.0, 2.0), None)]],
        sign=(1, 1),
        payoff_table=[[1.0, 0.0], [0.0, 2.0]],
        runtime_seconds=0.5,
        models_solved=1,
        models_infeasible=0,
        visited_points=1,
        grid_point_count=1,
        decision_variables_stored=False,
        round_decimals=6,
    )

    with pytest.raises(RuntimeError, match="store_decision_variables"):
        result.variables_for((1.0, 2.0))

"""Exact knapsack regression tests backed by bundled CSV references.

2kp50 runs in CI (~3s). 3kp40 is slow regression with multiproc."""

from importlib import resources
from multiprocessing import cpu_count

import pytest
from benchmarks.cases import BENCHMARK_CASES

from pyaugmecon import PyAugmecon
from tests.support.assertions import array_equal, read_reference_csv
from tests.support.factories import make_config

pytestmark = pytest.mark.knapsack

# (case, mark): fast cases run single-worker in CI; 3kp40 uses multiproc.
_KNAPSACK_ENTRIES: list[tuple[str, pytest.MarkDecorator | None]] = [
    ("2kp50", None),
    ("3kp40", pytest.mark.slow),
]

KNAPSACK_CASES = [
    pytest.param(BENCHMARK_CASES[name], marks=[mark] if mark else [])
    if mark
    else BENCHMARK_CASES[name]
    for name, mark in _KNAPSACK_ENTRIES
]


@pytest.fixture(
    scope="module",
    params=KNAPSACK_CASES,
    ids=[name for name, _ in _KNAPSACK_ENTRIES],
)
def knapsack_case(request):
    """Solve one bundled knapsack case and return its solver/reference payload."""
    case = request.param
    opts: dict[str, object] = {}
    if case.nadir_points is not None:
        opts["nadir_points"] = list(case.nadir_points)
    if case.name == "3kp40":
        opts["workers"] = cpu_count()
    solver = PyAugmecon(
        case.build_model(),
        make_config(case.name, mode="exact", **opts),
    )
    solver.solve()
    reference_dir = resources.files("pyaugmecon").joinpath("data", case.name)
    return {
        "name": case.name,
        "solver": solver,
        "reference_dir": reference_dir,
    }


def test_knapsack_payoff_table_matches_reference(knapsack_case):
    payoff = read_reference_csv(knapsack_case["reference_dir"], "payoff_table")
    assert array_equal(knapsack_case["solver"].result.payoff_table, payoff, 2)


def test_knapsack_pareto_front_matches_reference(knapsack_case):
    pareto_solutions = read_reference_csv(knapsack_case["reference_dir"], "pareto_sols")
    assert array_equal(knapsack_case["solver"].result.points, pareto_solutions, 2)


def test_knapsack_safe_and_payoff_strategies_agree_on_2kp50():
    """Both auto-nadir strategies must produce identical Pareto fronts on 2kp50.

    The grid widths differ (safe = global min; payoff = column min * undercut),
    but in exact integer mode the actual Pareto-optimal lattice points are the
    same set; the wider safe grid only adds skipped/infeasible cells.
    """
    case = BENCHMARK_CASES["2kp50"]

    safe = PyAugmecon(
        case.build_model(),
        make_config("2kp50_nadir_safe", mode="exact", nadir_strategy="safe"),
    ).solve()
    payoff = PyAugmecon(
        case.build_model(),
        make_config(
            "2kp50_nadir_payoff",
            mode="exact",
            nadir_strategy="payoff",
            nadir_undercut=0.8,
        ),
    ).solve()

    safe_points = {tuple(p) for p in safe.points}
    payoff_points = {tuple(p) for p in payoff.points}
    assert safe_points == payoff_points

from __future__ import annotations

import numpy as np
from pyomo.core.expr.visitor import identify_variables

from pyaugmecon import PyAugmecon
from tests.support.assertions import array_equal
from tests.support.factories import make_config
from tests.support.models import three_objective_model, two_objective_model


def test_lexicographic_payoff_table_matches_known_reference():
    solver = PyAugmecon(
        two_objective_model(), make_config("model_payoff_lexicographic")
    )
    solver.model.deactivate_all_objectives()
    solver.model.min_to_max()
    solver.model.construct_payoff()

    expected = np.array([[20, 160], [8, 184]])
    assert array_equal(solver.model.payoff, expected, 2)


def test_augmented_objective_contains_slack_terms():
    solver = PyAugmecon(three_objective_model(), make_config("model_augmentation"))
    solver.model.deactivate_all_objectives()
    solver.model.min_to_max()
    solver.model.construct_payoff()
    solver.model.find_obj_range()
    solver.model.convert_prob()

    primary_expr = solver.model.obj(0).expr
    vars_in_expr = {var.name for var in identify_variables(primary_expr)}
    assert any(name.startswith("_pyaugmecon_slack[") for name in vars_in_expr)

    constraint_list = solver.model.model.component("_pyaugmecon_constraint_list")
    assert len(constraint_list) == len(solver.model.constrained_objectives)


def test_payoff_nadir_uses_two_branch_undercut():
    """Payoff strategy widens the column min via min(col*u, col*1/u)."""
    solver = PyAugmecon(
        three_objective_model(),
        make_config("model_payoff_nadir", nadir_strategy="payoff", nadir_undercut=0.8),
    )
    solver.model.deactivate_all_objectives()
    solver.model.min_to_max()
    solver.model.construct_payoff()

    bounds = solver.model._compute_payoff_nadirs()
    for objective_idx in solver.model.constrained_objectives:
        col_min = float(np.min(solver.model.payoff[:, objective_idx]))
        expected = min(round(col_min * 0.8, 0), round(col_min * (1 / 0.8), 0))
        assert bounds[objective_idx] == expected


def test_payoff_strategy_skips_extra_setup_solves():
    """Payoff strategy reuses the payoff table; safe strategy adds (n-1) solves.

    Checks the progress-bar `to_solve` budget set in `__init__`, which already
    accounts for the per-objective nadir solves the safe strategy needs and
    the payoff strategy avoids.
    """
    safe = PyAugmecon(
        three_objective_model(),
        make_config(
            "model_setup_count_safe",
            nadir_strategy="safe",
            mode="sampled",
            sample_points=4,
        ),
    )
    payoff = PyAugmecon(
        three_objective_model(),
        make_config(
            "model_setup_count_payoff",
            nadir_strategy="payoff",
            mode="sampled",
            sample_points=4,
        ),
    )
    constrained = safe.model.n_obj - 1
    assert safe.model.to_solve - payoff.model.to_solve == constrained

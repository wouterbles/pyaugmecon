import numpy as np

from pyaugmecon import PyAugmecon
from tests.helper import Helper
from tests.optimization_models import two_objective_model

model_type = "two_objective_model"

options = {
    "name": model_type,
    "grid_points": 10,
}

py_augmecon = PyAugmecon(two_objective_model(), options)
py_augmecon.solve()


def test_payoff_table():
    payoff = np.array([[20, 160], [8, 184]])
    assert Helper.array_equal(py_augmecon.model.payoff, payoff, 2)


def test_pareto_sols():
    pareto_sols = np.array(
        [
            [8, 184],
            [9.33, 181.33],
            [10.67, 178.67],
            [12, 176],
            [13.33, 173.33],
            [14.67, 170.67],
            [16, 168],
            [17.33, 165.33],
            [18.67, 162.67],
            [20, 160],
        ]
    )
    assert Helper.array_equal(py_augmecon.unique_pareto_sols, pareto_sols, 2)

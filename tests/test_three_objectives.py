import numpy as np

from pyaugmecon import PyAugmecon
from tests.helper import Helper
from tests.optimization_models import three_objective_model

model_type = "three_objective_model"

options = {
    "name": model_type,
    "grid_points": 10,
}

py_augmecon = PyAugmecon(three_objective_model(), options)
py_augmecon.solve()


def test_payoff_table():
    payoff = np.array([[3075000, 62460, 33000], [3855000, 45180, 37000], [3225000, 55260, 23000]])
    assert Helper.array_equal(py_augmecon.model.payoff, payoff, 2)


def test_pareto_sols():
    pareto_sols = np.array(
        [
            [3075000, 62460, 33000],
            [3085000, 61980, 32333.33],
            [3108333.33, 60860, 30777.78],
            [3115000, 60540, 30333.33],
            [3131666.67, 59740, 29222.22],
            [3155000, 58620, 27666.67],
            [3178333.33, 57500, 26111.11],
            [3195000, 56700, 25000],
            [3201666.67, 56380, 24555.56],
            [3225000, 55260, 23000],
            [3255000, 54780, 23666.67],
            [3375000, 52860, 26333.33],
            [3495000, 50940, 29000],
            [3615000, 49020, 31666.67],
            [3735000, 47100, 34333.33],
            [3855000, 45180, 37000],
        ]
    )
    assert Helper.array_equal(py_augmecon.unique_pareto_sols, pareto_sols, 2)

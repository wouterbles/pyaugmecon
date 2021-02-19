from tests.helper import Helper
import numpy as np
from tests.optimization_models import two_objective_model
from pyaugmecon import *

moop_opts = {'grid_points': 10, 'early_exit': True}
solver_opts = {'solver_name': 'gurobi', 'solver_io': 'python'}
py_augmecon = MOOP(two_objective_model(), moop_opts, solver_opts)


def test_payoff_table():
    payoff_table = np.array([[20, 160], [8, 184]])
    assert Helper.array_equal(py_augmecon.payoff_table, payoff_table, 2)


def test_e_points():
    e_points = np.array([[
        160, 162.667, 165.333, 168, 170.667, 173.333, 176, 178.667, 181.333,
        184]])
    assert Helper.array_equal(py_augmecon.e, e_points, 2)


def test_pareto_sols():
    pareto_sols = np.array([
        [8, 184],
        [9.33, 181.33],
        [10.67, 178.67],
        [12, 176],
        [13.33, 173.33],
        [14.67, 170.67],
        [16, 168],
        [17.33, 165.33],
        [18.67, 162.67],
        [20, 160]])
    assert Helper.array_equal(py_augmecon.pareto_sols, pareto_sols, 2)

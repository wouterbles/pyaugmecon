from tests.helper import Helper
import numpy as np
import pandas as pd
from tests.optimizationModels import knapsack_model
from pyaugmecon import *

moop_opts = {'grid_points': 823, 'early_exit': True}
solver_opts = {'solver_name': 'gurobi', 'solver_io': 'python'}
py_augmecon = MOOP(knapsack_model('2kp50'), moop_opts, solver_opts)
xlsx = pd.ExcelFile(f"tests/input/2kp50.xlsx")


def test_payoff_table():
    payoff_table = pd.read_excel(xlsx, index_col=0, sheet_name='payoff_table').to_numpy()
    assert Helper.array_equal(py_augmecon.payoff_table, payoff_table, 2)


def test_e_points():
    e_points = pd.read_excel(xlsx, index_col=0, sheet_name='e_points').to_numpy()
    assert Helper.array_equal(py_augmecon.e, e_points, 2)


def test_pareto_sols():
    pareto_sols = pd.read_excel(xlsx, index_col=0, sheet_name='pareto_sols').to_numpy()
    assert Helper.array_equal(py_augmecon.pareto_sols, pareto_sols, 2)

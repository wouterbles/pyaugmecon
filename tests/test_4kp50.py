from tests.helper import Helper
import numpy as np
import pandas as pd
from tests.optimization_models import four_kp_model
from pyaugmecon import *

moop_opts = {
    'grid_points': 53,
    'nadir_points': [718, 717, 705],
    'early_exit': True,  # AUGMECON
    'bypass_coefficient': True,  # AUGMECON2
    'maximize': True,
    }

solver_opts = {
    'solver_name': 'gurobi',
    'solver_io': 'python',
    }

model_type = '4kp50'
py_augmecon = MOOP(
    four_kp_model(model_type),
    moop_opts,
    solver_opts,
    f'test_{model_type}'
    )

xlsx = pd.ExcelFile(f"tests/input/{model_type}.xlsx")


def test_payoff_table():
    payoff_table = Helper.read_excel(xlsx, 'payoff_table')
    assert Helper.array_equal(py_augmecon.payoff_table, payoff_table, 2)


def test_e_points():
    e_points = Helper.read_excel(xlsx, 'e_points')
    assert Helper.array_equal(py_augmecon.e, e_points, 2)


def test_pareto_sols():
    pareto_sols = Helper.read_excel(xlsx, 'pareto_sols')
    assert Helper.array_equal(py_augmecon.pareto_sols, pareto_sols, 2)

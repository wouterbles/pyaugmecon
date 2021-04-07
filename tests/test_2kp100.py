from tests.helper import Helper
import numpy as np
import pandas as pd
from tests.optimization_models import two_kp_model
from pyaugmecon import *

options = {
    'grid_points': 823,
    'early_exit': True,
    'bypass_coefficient': True,
    'maximize': True,
    }

model_type = '2kp100'
py_augmecon = MOOP(
    two_kp_model(model_type),
    options,
    f'test_{model_type}')

xlsx = pd.ExcelFile(f"tests/input/{model_type}.xlsx")


def test_payoff_table():
    payoff_table = Helper.read_excel(xlsx, 'payoff_table').to_numpy()
    assert Helper.array_equal(py_augmecon.payoff_table, payoff_table, 2)


def test_e_points():
    e_points = Helper.read_excel(xlsx, 'e_points').to_numpy()
    assert Helper.array_equal(py_augmecon.e, e_points, 2)


def test_pareto_sols():
    pareto_sols = Helper.read_excel(xlsx, 'pareto_sols').to_numpy()
    assert Helper.array_equal(py_augmecon.pareto_sols, pareto_sols, 2)

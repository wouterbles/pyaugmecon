import pandas as pd
from .helper import Helper
from pyaugmecon import PyAugmecon
from .optimization_models import four_kp_model

model_type = '4kp40'

options = {
    'name': model_type,
    'grid_points': 301,
    'nadir_points': [155, 119, 121],
    }

py_augmecon = PyAugmecon(four_kp_model(model_type), options)
py_augmecon.solve()

xlsx = pd.ExcelFile(f'tests/input/{model_type}.xlsx', engine='openpyxl')


def test_payoff_table():
    payoff = Helper.read_excel(xlsx, 'payoff_table').to_numpy()
    assert Helper.array_equal(py_augmecon.model.payoff, payoff, 2)


def test_pareto_sols():
    pareto_sols = Helper.read_excel(xlsx, 'pareto_sols').to_numpy()
    assert Helper.array_equal(py_augmecon.unique_pareto_sols, pareto_sols, 2)

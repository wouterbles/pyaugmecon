import pandas as pd

from pyaugmecon import PyAugmecon
from tests.helper import Helper
from tests.optimization_models import two_kp_model

model_type = "2kp250"

options = {
    "name": model_type,
    "grid_points": 2534,
}

py_augmecon = PyAugmecon(two_kp_model(model_type), options)
py_augmecon.solve()

xlsx = pd.ExcelFile(f"tests/input/{model_type}.xlsx", engine="openpyxl")


def test_payoff_table():
    payoff = Helper.read_excel(xlsx, "payoff_table").to_numpy()
    assert Helper.array_equal(py_augmecon.model.payoff, payoff, 2)


def test_pareto_sols():
    pareto_sols = Helper.read_excel(xlsx, "pareto_sols").to_numpy()
    assert Helper.array_equal(py_augmecon.unique_pareto_sols, pareto_sols, 2)

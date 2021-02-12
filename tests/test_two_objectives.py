from tests.helper import Helper
import numpy as np
from tests.optimizationModels import twoObjectiveModel
from pyaugmecon import *

MOOPopts = {'gridPoints': 10, 'earlyExit': True}
solverOpts = {'solverName': 'gurobi', 'solverIO': 'python'}
pyAugmecon = MOOP(twoObjectiveModel(), MOOPopts, solverOpts)


def test_payoff_table():
    payoff_table = np.array([[20, 160], [8, 184]])
    assert Helper.array_equal(pyAugmecon.payOffTable, payoff_table, 2)


def test_e_points():
    e_points = np.array([[
        160, 162.667, 165.333, 168, 170.667, 173.333, 176, 178.667, 181.333,
        184]])
    assert Helper.array_equal(pyAugmecon.e, e_points, 2)


def test_pareto_sols():
    paretoSols = np.array([
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
    assert Helper.array_equal(pyAugmecon.paretoSols, paretoSols, 2)

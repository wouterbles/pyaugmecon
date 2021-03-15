import os
import numpy as np
import pandas as pd
from pyaugmecon import *
from tests.optimization_models import (
    four_kp_model, three_kp_model, two_kp_model, three_objective_model,
    two_objective_model)

moop_opts = {
    'grid_points': 301,
    'nadir_points': [1031, 1069],
    'early_exit': True,  # AUGMECON
    'bypass_coefficient': True,  # AUGMECON2
    'maximize': True,
    }

solver_opts = {
    'solver_name': 'gurobi',
    'solver_io': 'python',
    }

A = MOOP(four_kp_model('4kp40'), moop_opts, solver_opts, '4kp40')
print('--- PAY-OFF TABLE ---')
print(A.payoff_table)
print('--')
print(A.ideal_point)
print('--- E-POINTS ---')
print(np.sort(A.e))
print(A.e.shape)
print('--')
print(A.obj_range)
print('--- PARETO SOLS ---')
print(A.pareto_sols.shape)
print('--')
print(A.pareto_sols)
print('--- MODELS SOLVED ---')
print(A.models_solved)

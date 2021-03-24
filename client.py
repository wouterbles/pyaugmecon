import os
import numpy as np
import pandas as pd
from pyaugmecon import *
from tests.optimization_models import (
    four_kp_model, three_kp_model, two_kp_model, three_objective_model,
    two_objective_model)

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
A = MOOP(four_kp_model(model_type), moop_opts, solver_opts, model_type)
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

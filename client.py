import numpy as np
from pyaugmecon import *
from tests.optimization_models import (
    economic_dispatch_model, knapsack_model, three_objective_model,
    two_objective_model)

moop_opts = {'grid_points': 492, 'early_exit': True}
solver_opts = {'solver_name': 'gurobi', 'solver_io': 'python'}

A = MOOP(knapsack_model("2kp50"), moop_opts, solver_opts)
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

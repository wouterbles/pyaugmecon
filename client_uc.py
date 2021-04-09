import numpy as np
import matplotlib.pyplot as plt
from pyaugmecon.pyaugmecon import MOOP
from tests.optimization_models import (unit_commitment_model)

options = {
    'grid_points': 10,
    'early_exit': True,  # AUGMECON
    'bypass_coefficient': True,  # AUGMECON2
    'flag_array': True,  # AUGMECON-R
    }

model_type = 'unit_commitment'
A = MOOP(unit_commitment_model(), options, model_type)
print('--- PAY-OFF TABLE ---')
print(A.payoff_table)
print('--')
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

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(A.pareto_sols[:, 0], A.pareto_sols[:, 1]), A.pareto_sols[:, 2]

ax.set(xlabel='Cost', ylabel='Emissions', zlabel='Unmet power')
ax.grid()
fig.savefig('sol.png')

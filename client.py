import pandas as pd
import numpy as np
from pyaugmecon.pyaugmecon import PyAugmecon
from tests.optimization_models import (
    unit_commitment_model, four_kp_model, three_kp_model, two_kp_model,
    three_objective_model, two_objective_model)

if __name__ == '__main__':
    model_type = '3kp40'

    options = {
        'name': model_type,
        'grid_points': 1540,
        'nadir_points': [31, 69],
        'early_exit': True,  # AUGMECON
        'bypass_coefficient': True,  # AUGMECON2
        'flag_array': True,  # AUGMECON-R
        #'nadir_ratio': 0.05,
        'redivide_work': False,
        'round_decimals': 0,
        'cpu_count': 1,
        }

    solver_options = {
        # 'Threads': 1,
    }

    A = PyAugmecon(three_kp_model(model_type), options, solver_options)
    A.solve()
    
    print('--- PAY-OFF TABLE ---')
    print(A.model.payoff)
    print('--- E-POINTS ---')
    print(np.sort(A.model.e))
    print('--')
    print(A.model.obj_range)
    print('--- PARETO SOLS ---')
    print(A.pareto_sols.shape)
    print('--')
    print(A.pareto_sols)
    print('--- MODELS SOLVED ---')
    print(A.model.models_solved.value())
    print('--- INFEASIBILITIES ---')
    print(A.model.infeasibilities.value())

    pd.DataFrame(A.pareto_sols).to_excel(
        f'{A.opts.logdir}/{A.opts.name}.xlsx')

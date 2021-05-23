import pandas as pd
import numpy as np
from pyaugmecon.pyaugmecon import PyAugmecon
from tests.optimization_models import (
    unit_commitment_model, four_kp_model, three_kp_model, two_kp_model,
    three_objective_model, two_objective_model)

if __name__ == '__main__':
    model_type = 'three'

    options = {
        'name': model_type,
        'grid_points': 10,
        # 'nadir_points': [1031, 1069],
        'early_exit': True,  # AUGMECON
        'bypass_coefficient': False,  # AUGMECON2
        'flag_array': False,  # AUGMECON-R
        'shared_flag': True,
        #'nadir_ratio': 0.9,
        # 'redivide_work': True,
        #'round_decimals': 0,
        'cpu_count': 1,
        }

    solver_options = {
        # 'Threads': 1,
    }

    A = PyAugmecon(three_objective_model(), options, solver_options)
    A.solve()

    print('--- PAY-OFF TABLE ---')
    print(A.model.payoff)
    #print('--- E-POINTS ---')
    #print(np.sort(A.model.e))
    #print('--')
    #print(A.model.obj_range)
    #print('--- PARETO SOLS ---')
    #print(A.pareto_sols.shape)
    #print('--')
    #print(A.pareto_sols)
    #print('--- MODELS SOLVED ---')
    #print(A.model.models_solved.value())
    #print('--- INFEASIBILITIES ---')
    #print(A.model.infeasibilities.value())

    writer = pd.ExcelWriter(f'{A.opts.logdir}/{A.opts.name}.xlsx')
    pd.DataFrame(A.pareto_sols).to_excel(writer, 'pareto_sols')
    pd.DataFrame(A.model.e).to_excel(writer, 'e_points')
    pd.DataFrame(A.model.payoff).to_excel(writer, 'payoff_table')
    writer.save()

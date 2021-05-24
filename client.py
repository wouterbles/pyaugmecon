import pandas as pd
import numpy as np
from pyaugmecon.pyaugmecon import PyAugmecon
from tests.optimization_models import (
    unit_commitment_model, four_kp_model, three_kp_model, two_kp_model,
    three_objective_model, two_objective_model)

if __name__ == '__main__':
    model_type = '2kp250'

    options = {
        'name': model_type,
        'grid_points': 2534,
        # 'nadir_points': [1031, 1069],
        'early_exit': True,  # AUGMECON
        'bypass_coefficient': True,  # AUGMECON2
        'flag_array': True,  # AUGMECON-R
        'shared_flag': True,
        # 'nadir_ratio': 0.9,
        # 'redivide_work': True,
        # 'round_decimals': 0,
        # 'cpu_count': 1,
        }

    solver_options = {
        # 'Threads': 1,
    }

    A = PyAugmecon(two_kp_model(model_type), options, solver_options)
    A.solve()

    writer = pd.ExcelWriter(f'{A.opts.logdir}/{A.opts.name}.xlsx')
    pd.DataFrame(A.sols).to_excel(writer, 'sols')
    pd.DataFrame(A.unique_sols).to_excel(writer, 'unique_sols')
    pd.DataFrame(A.unique_pareto_sols).to_excel(writer, 'unique_pareto_sols')
    pd.DataFrame(A.model.e).to_excel(writer, 'e_points')
    pd.DataFrame(A.model.payoff).to_excel(writer, 'payoff_table')
    writer.save()

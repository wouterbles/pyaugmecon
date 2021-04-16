import pandas as pd
from pyaugmecon.pyaugmecon import PyAugmecon
from tests.optimization_models import (
    unit_commitment_model, four_kp_model, three_kp_model, two_kp_model,
    three_objective_model, two_objective_model)

if __name__ == '__main__':
    model_type = 'uc_par_q'

    options = {
        'name': model_type,
        'grid_points': 20,
        #'nadir_points': [1031, 1069],
        'early_exit': True,  # AUGMECON
        'bypass_coefficient': True,  # AUGMECON2
        'flag_array': True,  # AUGMECON-R
        #'cpu_count': 4,
        }

    solver_options = {
        #'Threads': 1,
    }

    A = PyAugmecon(unit_commitment_model(), options, solver_options)
    A.solve()

    pd.DataFrame(A.pareto_sols).to_excel(
        f'{A.opts.logdir}/{A.opts.name}.xlsx')

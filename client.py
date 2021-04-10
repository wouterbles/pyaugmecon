from pyaugmecon.pyaugmecon import PyAugmecon
from tests.optimization_models import (
    unit_commitment_model, four_kp_model, three_kp_model, two_kp_model,
    three_objective_model, two_objective_model)

if __name__ == '__main__':
    model_type = '3kp40'

    options = {
        'name': model_type,
        'grid_points': 540,
        'nadir_points': [1031, 1069],
        'early_exit': True,  # AUGMECON
        'bypass_coefficient': True,  # AUGMECON2
        'flag_array': True,  # AUGMECON-R
        'cpu_count': 8,
        }

    solver_options = {
        'Threads': 1,
    }

    A = PyAugmecon(three_kp_model(model_type), options, solver_options)
    A.solve()

import numpy as np
from pyaugmecon import *
from tests.optimization_models import (
    unit_commitment_model, four_kp_model, three_kp_model, two_kp_model,
    three_objective_model, two_objective_model)

if __name__ == '__main__':
    options = {
        'grid_points': 823,
        #'nadir_points': [1031, 1069],
        'early_exit': True,  # AUGMECON
        'bypass_coefficient': True,  # AUGMECON2
        'flag_array': True,  # AUGMECON-R
        }

    model_type = '2kp100'
    A = MOOP(two_kp_model(model_type), options, model_type)

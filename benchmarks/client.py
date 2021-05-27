from pyaugmecon.pyaugmecon import PyAugmecon
from tests.optimization_models import three_kp_model

if __name__ == '__main__':
    model_type = '3kp40'

    options = {
        'name': model_type,
        'grid_points': 540,
        'nadir_points': [1031, 1069],
        }

    py_augmecon = PyAugmecon(three_kp_model(model_type), options)
    py_augmecon.solve()

from tests.optimization_models import (
    four_kp_model, three_kp_model, two_kp_model)


def model_data():
    return {
        '2kp50': {
            'model': two_kp_model,
            'opts': {
                'grid_points': 492,
            },
        },
        '2kp100': {
            'model': two_kp_model,
            'opts': {
                'grid_points': 823,
            },
        },
        '2kp250': {
            'model': two_kp_model,
            'opts': {
                'grid_points': 492,
            },
        },
        '3kp40': {
            'model': three_kp_model,
            'opts': {
                'grid_points': 540,
                'nadir_points': [1031, 1069],
            },
        },
        '3kp50': {
            'model': three_kp_model,
            'opts': {
                'grid_points': 847,
                'nadir_points': [1124, 1041],
            },
        },
        '4kp40': {
            'model': four_kp_model,
            'opts': {
                'grid_points': 301,
                'nadir_points': [155, 119, 121],
            },
        },
        '4kp50': {
            'model': four_kp_model,
            'opts': {
                'grid_points': 53,
                'nadir_points': [718, 717, 705],
            },
        },
    }

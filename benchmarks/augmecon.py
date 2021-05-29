from pyaugmecon.pyaugmecon import PyAugmecon
from benchmarks.model_data import model_data


def augmecon():
    data = model_data()
    general_opts = {
        "logging_folder": "benchmarks/logs_augmecon",
        "shared_flag": False,
        "redivide_work": False,
        "flag_array": False,
        "bypass_coefficient": False,
        "cpu_count": 1,
        "process_timeout": 3600 * 4,  # timeout after 4 hours
    }

    for model_name in data:
        model = data[model_name]["model"]
        opts = data[model_name]["opts"]
        opts.update(general_opts)
        opts["name"] = model_name

        py_augmecon = PyAugmecon(model(model_name), opts)
        py_augmecon.solve()

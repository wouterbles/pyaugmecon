from pyaugmecon.pyaugmecon import PyAugmecon
from benchmarks.model_data import model_data


def augmecon_r():
    data = model_data()
    general_opts = {
        "logging_folder": "benchmarks/logs_augmecon_r",
        "shared_flag": False,
        "redivide_work": False,
        "cpu_count": 1,
    }

    for model_name in data:
        model = data[model_name]["model"]
        opts = data[model_name]["opts"]
        opts.update(general_opts)
        opts["name"] = model_name

        py_augmecon = PyAugmecon(model(model_name), opts)
        py_augmecon.solve()

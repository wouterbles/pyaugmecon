from pyaugmecon.pyaugmecon import PyAugmecon
from benchmarks.model_data import model_data


def parallelization_default():
    data = model_data()
    general_opts = {
        "logging_folder": "benchmarks/logs_parallelization_default",
    }

    for model_name in data:
        model = data[model_name]["model"]
        opts = data[model_name]["opts"]
        opts.update(general_opts)
        opts["name"] = model_name

        py_augmecon = PyAugmecon(model(model_name), opts)
        py_augmecon.solve()

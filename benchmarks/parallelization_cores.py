from pyaugmecon.pyaugmecon import PyAugmecon
from benchmarks.model_data import model_data


def parallelization_cores():
    data = model_data()
    general_opts = {
        'logging_folder': 'benchmarks/logs_parallelization_cores',
    }

    # No parallelization for two-objective functions, so testing different
    # numbers of cores is not interesting
    to_remove = ('2kp50', '2kp100', '2kp250')
    for k in to_remove:
        data.pop(k, None)

    for model_name in data:
        for cores in range(1, 48, 2):
            model = data[model_name]['model']
            opts = data[model_name]['opts']
            general_opts['cpu_count'] = cores
            opts.update(general_opts)
            opts['name'] = model_name

            py_augmecon = PyAugmecon(model(model_name), opts)
            py_augmecon.solve()

# PyAUGMECON
This is a [Python](https://www.python.org/) and [Pyomo](http://www.pyomo.org/) based implementation of the augmented ε-constraint (AUGMECON) method and its variants which can be used to solve multi-objective optimization problems, i.e., to return an approximation of the exact Pareto front.

It currently supports:

- Inner loop early exit (AUGMECON)
- Bypass coefficient (AUGMECON2)
- Flag array (AUGMECON-R)
- Processs-based parallelization

[GAMS implementations](#useful-resources) of the method and its variants were provided by the authors of the papers. To the best of my knowledge, this is the first publicly available [Python](https://www.python.org/) implementation.

## Installation
Until this project is published on [PyPI](https://pypi.org/), more steps are needed to set it up.

### Requirements
- [Python 3.8](https://www.python.org/)
- [Pyomo](http://www.pyomo.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Cloudpickle](https://github.com/cloudpipe/cloudpickle)
- [Gurobi](https://www.gurobi.com/) (other solvers currently not supported)

### Anaconda installation (advised)
Only Gurobi and Pyomo need to be installed as other tools are by default included in Anaconda.
```
# Install Anaconda
wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash /tmp/Anaconda3-2020.11-Linux-x86_64.sh

# Install Gurobi
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi

# Install Pyomo
conda install -c conda-forge pyomo
```

## Usage
First, an optimization model generator function needs to be created (examples in tests folder) to pass the model to PyAUGMECON. This function should return an unsolved instance of the optimization problem. Essentially, the only difference in comparison with creating a single objective Pyomo optimization model is the fact that multiple objectives are defined using Pyomo's `ObjectiveList()`.

The rest of the process is automatically being taken care of by instantiating a `PyAugmecon()` object and solving it afterwards with `PyAugmecon().solve()`.

### PyAugmecon parameters
`PyAugmecon(model, opts, solver_opts)`

| Name                      | Description |
|---------------------------|-------------|
| `model`       | Function that returns an unsolved instance of the optimization problem, see [usage](#usage) 
| `opts`        | Dictionary of pyAUGMECON related options, see [PyAugmecon options](#pyaugmecon-options)    
| `solver_opts` | Dictionary of solver (Gurobi) options, see [solver options](#solver-options)

### Example 3kp40
```python
model_type = '3kp40'

options = {
    'name': model_type,
    'grid_points': 100, 
    'nadir_points': [1031, 1069],
    }

solver_options = {
    'Threads': 1,
}

A = PyAugmecon(three_kp_model(model_type), options, solver_options)
A.solve()
print(A.payoff_table) # this prints the payoff table
print(A.pareto_sols) # this prints the unique Pareto optimal solutions
```

### PyAugmecon object attributes
| Name                      | Description |
|---------------------------|-------------|
| `model.models_solved`     | Number of models solved (excluding solves for payoff table) |
| `model.model`             | Pyomo model isntance |
| `model.to_solve`          | Total number of models to solve (including payoff table) |
| `model.payoff`            | Payoff table |
| `model.e`                 | Gridpoints of p-1 objective functions that are used as constraints |
| `num_unique_pareto_sols`  | Number of unique Pareto solutions |
| `pareto_sols`             | Unique Pareto solutions |
| `runtime`                 | Total runtime in seconds |

### PyAugmecon options
| Option                    | Description     | Default |
|---------------------------|-----------------|---------|
| `name`                    | Name of the model, used in logging output | |
| `grid_points`             | Number of grid points (if X grid points in GAMS, X+1 grid points are needed in order to exactly replicate results) | |
| `nadir_points`            | If the exact nadir points are known, these can be provided, otherwise they will be taken from the payoff table  | `true` |
| `early_exit`              | Use inner loop early exit (AUGMECON), exit the inner loop when the model result becomes infeasible | `true` |
| `bypass_coefficient`      | Use bypass coefficient (AUGMECON2), utilize slack variables to skip redundant iterations  | `true` |
| `flag_array`              | Use flag array (AUGMECON-R), store early exit and bypass coefficients for upcoming iterations | `true` |
| `penalty_weight`          | The penalty by which other terms in the objective are multiplied, usually between `1e-3` and `1e-6` | `1e-3` |
| `nadir_ratio`             | For problems with three or more objective functions the payoff table minima do not guarantee the exact nadir. This factor scales the payoff minima. By providing a value lower than one, lower bounds of the minima can be estimated | `1` |
| `logging_folder`          | Folder to store log files (relative to root directory) | `logs` |
| `cpu_count`               | Specify over how many processes the work should be divided. Use `1` to disable parallelization | `multiprocessing.cpu_count()`|
| `redivide_work`           | Have processes take work from unfinished processes when their queue is empty | `true` |
| `pickle_file`             | File name of the pickled [Pyomo](http://www.pyomo.org/) model for [cloudpickle](https://github.com/cloudpipe/cloudpickle) | `model.p` |
| `solver_name`             | Name of the solver provided to [Pyomo](http://www.pyomo.org/) | `gurobi` |
| `solver_io`               | Name of the solver interface provided to [Pyomo](http://www.pyomo.org/) | `python` |

### Solver options
| Option                    | Description     | Default |
|---------------------------|-----------------|---------|
| `MIPGap`                  | See Gurobi documentation: [MIPGap](https://www.gurobi.com/documentation/9.1/refman/mipgap2.html) | `0.0` |
| `NonConvex`               | See Gurobi documentation [NonConvex](https://www.gurobi.com/documentation/9.1/refman/nonconvex.html) | `2` | 

## Tests
To test the correctness of the proposed approach, the following optimization models have been tested both using the GAMS and the proposed implementations:

- Sample bi-objective problem of the original paper
- Sample three-objective problem of the original implementation/presentation
- Multi-objective multidimensional knapsack (MOMKP) problems: 2kp50, 2kp100, 2kp250, 3kp40, 3kp50, 4kp40 and 4kp50
- Multi-objective economic dispatch (minimize cost, emissions and unmet demand)

These models can be found in the **tests** folder and run with [PyTest](https://pytest.org) to test for correctness of the payoff table and the Pareto solutions. Example: `pytest test/test_3kp40.py`

> Despite the extensive testing, the PyAUGMECON implementation is provided without any warranty. Use it at your own risk!

## Useful resources

### Original AUGMECON
- G. Mavrotas, “Effective implementation of the ε-constraint methodin Multi-Objective Mathematical Programming problems,”*Applied Mathematics and Computation*, vol. 213, no. 2, pp. 455–465, 2009
- [GAMS implementation](https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_epscm.html)

### Improved AUGMECON2
- Mavrotas and K. Florios, “An improved version of the augmented ε-constraint method (AUGMECON2) for finding the exact pareto set in multi-objective integer programming problems,” *Applied Mathematics and Computation*, vol. 219, no. 18, pp. 9652–9669, 2013.
- [GAMS implementation](https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_epscmmip.html)

### Further improved AUGMECON-R
- A. Nikas, A. Fountoulakis, A. Forouli,and H. Doukas, *A robust augmented εconstraint method (AUGMECON-R) for finding exact solutions of multi-objective linear programming problems.* Springer Berlin Heidelberg, 2020, no. 0123456789.
- [GAMS implementation](https://github.com/KatforEpu/Augmecon-R)

### Other resources
- The following PhD thesis is also very useful: \
https://www.chemeng.ntua.gr/gm/gmsite_gre/index_files/PHD_mavrotas_text.pdf
- This presentation provides a quick overview: \
https://www.chemeng.ntua.gr/gm/gmsite_eng/index_files/mavrotas_MCDA64_2006.pdf

## Notes

- The choice of the objective function(s) to add as constraints affects the mapping of the Pareto front. Empirically, having one of the objective functions with the large range in the constraint set tends to result in a denser representation of the Pareto front. Nevertheless, despite the choice of which objectives to add in the constraint set, the solutions belong to the same Pareto front (obviously).

- If X grid points in GAMS, X+1 grid points are needed in PyAUGMECON in order to exactly replicate results

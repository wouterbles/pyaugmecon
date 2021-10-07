from pyaugmecon import PyAugmecon
from tests.optimization_models import three_kp_model

# Multiprocessing requires this If statement (on Windows)
if __name__ == "__main__":
    model_type = "3kp40"

    # AUGMECON related options
    opts = {
        "name": model_type,
        "grid_points": 540,
        "nadir_points": [1031, 1069],
    }

    # Options passed to Gurobi
    solver_opts = {}

    A = PyAugmecon(
        three_kp_model(model_type), opts, solver_opts
    )  # instantiate  PyAugmecon
    A.solve()  # solve PyAugmecon multi-objective optimization problem
    print(A.model.payoff)  # this prints the payoff table
    print(A.unique_pareto_sols)  # this prints the unique Pareto optimal solutions

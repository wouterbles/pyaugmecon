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

    pyaugmecon = PyAugmecon(three_kp_model(model_type), opts)  # instantiate  PyAugmecon
    pyaugmecon.solve()  # solve PyAugmecon multi-objective optimization problem
    sols = pyaugmecon.get_pareto_solutions()  # get all pareto solutions
    payoff = pyaugmecon.get_payoff_table()  # get the payoff table
    decision_vars = pyaugmecon.get_decision_variables(sols[0])  # get the decision variables

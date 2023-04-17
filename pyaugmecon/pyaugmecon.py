import itertools
import logging
from typing import List

import numpy as np
import pandas as pd
from pymoo.config import Config
from pymoo.indicators.hv import HV
from pyomo.environ import Model as PyomoModel

from pyaugmecon.helper import Helper, Timer
from pyaugmecon.logs import Logs
from pyaugmecon.model import Model
from pyaugmecon.options import Options
from pyaugmecon.process_handler import ProcessHandler
from pyaugmecon.queue_handler import QueueHandler


class PyAugmecon:
    def __init__(self, model: PyomoModel, opts: Options, solver_opts={}):
        """
        Initialize a PyAugmecon object.

        Parameters
        ----------
        model : PyomoModel
            The optimization problem to solve.
        opts : Options
            Options for the PyAugmecon solver.
        solver_opts : dict, optional
            Solver-specific options. Default is an empty dictionary.
        """
        # Initialize Options and Logs objects with the given options
        self.opts = Options(opts, solver_opts)
        self.logs = Logs(self.opts)

        # Initialize logger and set log level to INFO
        self.logger = logging.getLogger(self.opts.log_name)
        self.logger.setLevel(logging.INFO)

        # Log options and initialize Model object with the given model
        self.opts.log()
        self.model = Model(model, self.opts)

        self.opts.check(self.model.n_obj)  # Check the number of objectives against the given options
        Config.warnings["not_compiled"] = False  # Suppress pymoo warnings

        # Initialize solutions
        self.sols = None
        self.unique_sols = None
        self.unique_pareto_sols = None

    def _find_solutions(self):
        """
        Find solutions to the optimization problem using the AUGMECON method.

        """
        # Set progress message
        self.model.progress.set_message("finding solutions")

        # Generate grid of indices to search for solutions
        grid_range = range(self.opts.gp)
        indices = [tuple([n for n in grid_range]) for _ in self.model.iter_obj2]
        self.cp = list(itertools.product(*indices))
        self.cp = [i[::-1] for i in self.cp]

        # Pickle the model and initialize queue and process handlers
        self.model.pickle()
        self.queues = QueueHandler(self.cp, self.opts)
        self.queues.split_work()
        self.procs = ProcessHandler(self.opts, self.model, self.queues)

        # Start processes and wait for results
        self.procs.start()
        self.unprocesssed_sols = self.queues.get_result()
        self.procs.join()

        # Clean the pickled model
        self.model.clean()

    def _process_solutions(self):
        def convert_obj_goal(sols: np.ndarray):
            return np.array(sols) * self.model.obj_goal

        def convert_obj_goal_dict(sols: dict):
            return {(tuple(x * y for x, y in zip(key, self.model.obj_goal))): sols[key] for key in sols}

        def keep_undominated(pts):
            pts = np.array(pts)
            undominated = np.ones(pts.shape[0], dtype=bool)
            for i, c in enumerate(pts):
                if undominated[i]:
                    undominated[undominated] = np.any(pts[undominated] > c, axis=1)
                    undominated[i] = True

            return pts[undominated, :]

        # Merge solutions into one dictionary and remove duplicates
        self.sols = {}
        for sol in self.unprocesssed_sols:
            self.sols.update(sol)
        self.num_sols = len(self.sols)

        # Remove duplicate solutions due to numerical issues by rounding
        self.unique_sols = {
            tuple(round(val, self.opts.round) for val in key): value for key, value in self.sols.items()
        }
        self.num_unique_sols = len(self.unique_sols)

        # Remove dominated solutions
        unique_pareto_keys = keep_undominated(list(self.unique_sols.keys()))
        unique_pareto_keys = [tuple(subarr) for subarr in unique_pareto_keys]
        self.unique_pareto_sols = {k: self.unique_sols[k] for k in unique_pareto_keys if k in self.unique_sols}
        self.num_unique_pareto_sols = len(self.unique_pareto_sols)

        # Multiply by -1 if original objective was minimization
        self.model.payoff = convert_obj_goal(self.model.payoff)
        self.sols = convert_obj_goal_dict(self.sols)
        self.unique_sols = convert_obj_goal_dict(self.unique_sols)
        self.unique_pareto_sols = convert_obj_goal_dict(self.unique_pareto_sols)

    def _output_excel(self):
        """
        Save the model's data to an Excel file.

        The data includes the `e` points, payoff table, solutions,
        unique solutions, and unique Pareto solutions.
        """
        # Create an Excel writer object to write the data to
        writer = pd.ExcelWriter(f"{self.logs.logdir}{self.opts.log_name}.xlsx")

        # Write the data to the sheets in the Excel file
        pd.DataFrame(self.model.e).to_excel(writer, "e_points")
        pd.DataFrame(self.model.payoff).to_excel(writer, "payoff_table")
        pd.DataFrame(Helper.keys_to_list(self.sols)).to_excel(writer, "sols")
        pd.DataFrame(Helper.keys_to_list(self.unique_sols)).to_excel(writer, "unique_sols")
        pd.DataFrame(Helper.keys_to_list(self.unique_pareto_sols)).to_excel(writer, "unique_pareto_sols")

        # Close the Excel writer object
        writer.close()

    def _get_hv_indicator(self):
        """
        Compute the hypervolume (HV) indicator for the unique Pareto solutions.

        The HV indicator measures the volume of the dominated space below the
        reference point, which is defined as the diagonal of the payoff table.

        """
        # Define the reference point as the diagonal of the payoff table
        ref = np.diag(self.model.payoff)

        # Create an HV object with the reference point
        ind = HV(ref_point=ref)

        # Compute the HV indicator for the unique Pareto solutions and store it
        # in the `hv_indicator` attribute
        self.hv_indicator = ind(np.array(Helper.keys_to_list(self.unique_pareto_sols)))

    def get_pareto_solutions(self) -> List[tuple]:
        """
        Get a list of Pareto-optimal solutions.

        Returns
        -------
        pareto_solutions : list[tuple]
            List of Pareto-optimal solutions.

        """
        return list(self.unique_pareto_sols.keys())

    def get_decision_variables(self, pareto_solution: tuple) -> dict:
        """
        Get a dictionary of decision variables for a given Pareto-optimal solution.

        Parameters
        ----------
        pareto_solution : tuple
            Tuple representing a Pareto-optimal solution.

        Returns
        -------
        decision_vars : dict
            Dcitionary of decision variables for a given Pareto-optimal solution. Where the key represents the decision
            variable name and the value is a pd.Series with the values.

        """
        # If a Pareto-optimal solution is provided as an argument, check if it exists in the dictionary
        if pareto_solution not in self.unique_pareto_sols:
            raise ValueError(f"Pareto solution not found: {pareto_solution}")
        return self.unique_pareto_sols[pareto_solution]

    def get_payoff_table(self) -> np.ndarray:
        """
        Get the payoff table from the model.

        Returns
        -------
        payoff_table : ndarray
            2-D array containing the payoff values for each combination of objectives.

        """
        return self.model.payoff

    def solve(self):
        """
        Solve the optimization problem and save the results.
        """
        self.runtime = Timer()  # Start the timer to measure the runtime
        self.model.min_to_max()  # Convert minimization problems to maximization problems
        self.model.construct_payoff()  # Construct a payoff table from the objective function values
        self.model.find_obj_range()  # Find the range of each objective function
        self.model.convert_prob()  # Convert the payoff table

        self._find_solutions()  # Find all solutions to the optimization problem
        self._process_solutions()  # Identify the unique solutions
        self._get_hv_indicator()  # Compute the HV indicator

        # Save the results to an Excel file if requested
        if self.opts.output_excel:
            self._output_excel()

        # Compute the total runtime and print a summary of the results
        self.runtime = round(self.runtime.get(), 2)
        Helper.clear_line()
        print(
            f"Solved {self.model.models_solved.value()} models for "
            f"{self.num_unique_pareto_sols} unique Pareto solutions in "
            f"{self.runtime} seconds"
        )

        # Log a summary of the results
        self.logger.info(Helper.separator())
        self.logger.info(f"Runtime: {self.runtime} seconds")
        self.logger.info(f"Models solved: {self.model.models_solved.value()}")
        self.logger.info(f"Infeasibilities: {self.model.infeasibilities.value()}")
        self.logger.info(f"Solutions: {self.num_sols}")
        self.logger.info(f"Unique solutions: {self.num_unique_sols}")
        self.logger.info(f"Unique Pareto solutions: {self.num_unique_pareto_sols}")
        self.logger.info(f"Hypervolume indicator: {self.hv_indicator}")

import logging
import time
from multiprocessing import cpu_count

from pyaugmecon.helper import Helper


class Options:
    def __init__(self, opts: dict, solver_opts: dict):
        """
        Initialize options with default values, and override them with user-defined options.

        Parameters
        ----------
        opts : dict
            A dictionary containing user-defined options.
        solver_opts : dict
            A dictionary containing solver options.

        """
        self.name = opts.get("name", "Undefined")  # Name of the problem
        self.gp = opts.get("grid_points")  # Number of grid points
        self.nadir_p = opts.get("nadir_points")  # Nadir points
        self.eps = opts.get("penalty_weight", 1e-3)  # Penalty weight
        self.round = opts.get("round_decimals", 9)  # Decimal places to round to
        self.nadir_r = opts.get("nadir_ratio", 1)  # Nadir ratio
        self.logdir = opts.get("logging_folder", "logs")  # Folder to save logs
        self.early_exit = opts.get("early_exit", True)  # Whether to enable early exit
        self.bypass = opts.get("bypass_coefficient", True)  # Whether to enable bypass coefficient
        self.flag = opts.get("flag_array", True)  # Whether to use flag array
        self.cpu_count = opts.get("cpu_count", cpu_count())  # Number of CPUs to use
        self.redivide_work = opts.get("redivide_work", True)  # Whether to redivide work
        self.model_fn = opts.get("pickle_file", "model.p")  # Pickle file name
        self.shared_flag = opts.get("shared_flag", True)  # Whether to use shared flag array
        self.output_excel = opts.get("output_excel", True)  # Whether to output to Excel
        self.process_logging = opts.get("process_logging", False)  # Whether to enable process logging
        self.process_timeout = opts.get("process_timeout", None)  # Timeout for processes
        self.solver_name = opts.get("solver_name", "gurobi")  # Name of solver
        self.solver_io = opts.get("solver_io", "python")  # IO mode of solver

        self.solver_opts = solver_opts  # Solver options
        self.solver_opts["MIPGap"] = solver_opts.get("MIPGap", 0.0)  # MIP gap
        self.solver_opts["NonConvex"] = solver_opts.get("NonConvex", 2)  # Nonconvex setting

        # Remove None values from dict when user has overriden them
        for key, value in dict(self.solver_opts).items():
            if value is None or value:
                del self.solver_opts[key]

        self.time_created = time.strftime("%Y%m%d-%H%M%S")  # Time the options object was created
        self.log_name = self.name + "_" + str(self.time_created)  # Name of log file

    def log(self):
        """
        Log the options using the logging module.
        """
        self.logger = logging.getLogger(self.log_name)
        self.logger.info(f"Name: {self.name}")
        self.logger.info(f"Grid points: {self.gp}")
        self.logger.info(f"Nadir points: {self.nadir_p}")
        self.logger.info(f"Penalty weight: {self.eps}")
        self.logger.info(f"Early exit: {self.early_exit}")
        self.logger.info(f"Bypass coefficient: {self.bypass}")
        self.logger.info(f"Flag array: {self.flag}")
        self.logger.info(f"CPU Count: {self.cpu_count}")
        self.logger.info(f"Redivide work: {self.redivide_work}")
        self.logger.info(f"Shared flag array: {self.shared_flag}")
        self.logger.info(Helper.separator())

    def check(self, num_objfun):
        """
        Check if the options are valid.

        Parameters
        ----------
        num_objfun : int
            The number of objective functions.

        Raises
        ------
        Exception
            If the number of grid points is not provided or the number of nadir points is too many or too few.

        """
        if not self.gp:
            raise Exception("No number of grid points provided")

        if self.nadir_p and len(self.nadir_p) != num_objfun - 1:
            raise Exception("Too many or too few nadir points provided")

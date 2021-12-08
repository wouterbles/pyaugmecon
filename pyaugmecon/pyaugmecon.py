import logging
import itertools
import numpy as np
import pandas as pd
from pymoo.factory import get_performance_indicator
from pyaugmecon.logs import Logs
from pyaugmecon.model import Model
from pyaugmecon.helper import Helper, Timer
from pyaugmecon.options import Options
from pyaugmecon.queue_handler import QueueHandler
from pyaugmecon.process_handler import ProcessHandler


class PyAugmecon(object):
    def __init__(self, model, opts, solver_opts={}):
        self.opts = Options(opts, solver_opts)
        self.logs = Logs(self.opts)
        self.logger = logging.getLogger(self.opts.log_name)
        self.logger.setLevel(logging.INFO)
        self.opts.log()
        self.model = Model(model, self.opts)
        self.opts.check(self.model.n_obj)

    def find_solutions(self):
        self.model.progress.set_message("finding solutions")

        grid_range = range(self.opts.gp)
        indices = [tuple([n for n in grid_range]) for _ in self.model.iter_obj2]
        self.cp = list(itertools.product(*indices))
        self.cp = [i[::-1] for i in self.cp]

        self.model.pickle()
        self.queues = QueueHandler(self.cp, self.opts)
        self.queues.split_work()
        self.procs = ProcessHandler(self.opts, self.model, self.queues)

        self.procs.start()
        self.unprocesssed_sols = self.queues.get_result()
        self.procs.join()
        self.model.clean()

    def process_solutions(self):
        def convert_obj_goal(sols):
            return np.array(sols) * self.model.obj_goal

        def keep_undominated(pts):
            pts = np.array(pts)
            undominated = np.ones(pts.shape[0], dtype=bool)
            for i, c in enumerate(pts):
                if undominated[i]:
                    undominated[undominated] = np.any(pts[undominated] > c, axis=1)
                    undominated[i] = True

            return pts[undominated, :]

        # Remove duplicate solutions
        self.sols = list(set(tuple(self.unprocesssed_sols)))
        self.sols = [list(i) for i in self.sols]
        self.num_sols = len(self.sols)

        # Remove duplicate solutions due to numerical issues by rounding
        self.unique_sols = [
            tuple(round(sol, self.opts.round) for sol in item) for item in self.sols
        ]
        self.unique_sols = list(set(tuple(self.unique_sols)))
        self.unique_sols = [list(i) for i in self.unique_sols]
        self.num_unique_sols = len(self.unique_sols)

        # Remove dominated solutions
        self.unique_pareto_sols = keep_undominated(self.unique_sols)
        self.num_unique_pareto_sols = len(self.unique_pareto_sols)

        # Multiply by -1 if original objective was minimization
        self.model.payoff = convert_obj_goal(self.model.payoff)
        self.sols = convert_obj_goal(self.sols)
        self.unique_sols = convert_obj_goal(self.unique_sols)
        self.unique_pareto_sols = convert_obj_goal(self.unique_pareto_sols)

    def output_excel(self):
        writer = pd.ExcelWriter(f"{self.logs.logdir}{self.opts.log_name}.xlsx")
        pd.DataFrame(self.model.e).to_excel(writer, "e_points")
        pd.DataFrame(self.model.payoff).to_excel(writer, "payoff_table")
        pd.DataFrame(self.sols).to_excel(writer, "sols")
        pd.DataFrame(self.unique_sols).to_excel(writer, "unique_sols")
        pd.DataFrame(self.unique_pareto_sols).to_excel(writer, "unique_pareto_sols")
        writer.save()

    def get_hv_indicator(self):
        ref = np.diag(self.model.payoff)
        hv = get_performance_indicator("hv", ref_point=ref)
        self.hv_indicator = hv.do(self.unique_pareto_sols)

    def solve(self):
        self.runtime = Timer()
        self.model.min_to_max()
        self.model.construct_payoff()
        self.model.find_obj_range()
        self.model.convert_prob()
        self.find_solutions()
        self.process_solutions()
        self.get_hv_indicator()
        if self.opts.output_excel:
            self.output_excel()

        self.runtime = round(self.runtime.get(), 2)
        Helper.clear_line()
        print(
            f"Solved {self.model.models_solved.value()} models for "
            f"{self.num_unique_pareto_sols} unique Pareto solutions in "
            f"{self.runtime} seconds"
        )

        self.logger.info(Helper.separator())
        self.logger.info(f"Runtime: {self.runtime} seconds")
        self.logger.info(f"Models solved: {self.model.models_solved.value()}")
        self.logger.info(f"Infeasibilities: {self.model.infeasibilities.value()}")
        self.logger.info(f"Solutions: {self.num_sols}")
        self.logger.info(f"Unique solutions: {self.num_unique_sols}")
        self.logger.info(f"Unique Pareto solutions: {self.num_unique_pareto_sols}")
        self.logger.info(f"Hypervolume indicator: {self.hv_indicator}")

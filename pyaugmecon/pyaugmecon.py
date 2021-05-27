import logging
import time
import itertools
import numpy as np
import pandas as pd
from pyaugmecon.flag import Flag
from pyaugmecon.logs import Logs
from pyaugmecon.model import Model
from pyaugmecon.helper import Helper
from pyaugmecon.options import Options
from pyaugmecon.queue_handler import QueueHandler
from pyaugmecon.process_handler import ProcessHandler


def solve_grid(
        pid,
        opts: Options,
        model: Model,
        queues: QueueHandler,
        flag: Flag):

    jump = 0
    pareto_sols = []
    if opts.process_logging:
        logger = logging.getLogger(opts.log_name)
        logger.setLevel(logging.INFO)

    model.unpickle()

    while True:
        work = queues.get_work(pid)

        if work:
            for c in work:
                log = f'PID: {pid}, index: {c}, '

                cp_start = opts.gp - 1 if model.min_obj else 0
                cp_end = 0 if model.min_obj else opts.gp - 1

                model.progress.increment()

                def do_jump(i, jump):
                    return min(jump, abs(cp_end - i))

                def bypass_range(i):
                    if i == 0:
                        return range(c[i], c[i] + 1)
                    elif model.min_obj:
                        return range(c[i] - b[i], c[i] + 1)
                    else:
                        return range(c[i], c[i] + b[i] + 1)

                def early_exit_range(i):
                    if i == 0:
                        return range(c[i], c[i] + 1)
                    elif model.min_obj:
                        return range(c[i], cp_start)
                    else:
                        return range(c[i], cp_end)

                if opts.flag and flag.get(c) != 0 and jump == 0:
                    jump = do_jump(c[0] - 1, flag.get(c))

                if jump > 0:
                    jump = jump - 1
                    continue

                for o in model.iter_obj2:
                    log += f'e{o + 1}: {model.e[o, c[o]]}, '
                    model.model.e[o + 2] = model.e[o, c[o]]

                model.obj_activate(0)
                model.solve()
                model.models_solved.increment()

                if (opts.early_exit and model.is_infeasible()):
                    model.infeasibilities.increment()
                    if opts.flag:
                        flag.set(early_exit_range, opts.gp, model.iter_obj2)
                    jump = do_jump(c[0], opts.gp)

                    log += 'infeasible'
                    if opts.process_logging:
                        logger.info(log)
                    continue
                elif (opts.bypass and
                      model.is_status_ok() and model.is_feasible()):
                    b = []

                    for i in model.iter_obj2:
                        step = model.obj_range[i] / (opts.gp - 1)
                        slack = round(model.slack_val(i + 1))
                        b.append(int(slack/step))

                    log += f'jump: {b[0]}, '

                    if opts.flag:
                        flag.set(bypass_range, b[0] + 1, model.iter_obj2)
                    jump = do_jump(c[0], b[0])

                tmp = []

                tmp.append(model.obj_val(0) - opts.eps
                           * sum(10**(-1*(o))*model.slack_val(o + 1)
                                 / model.obj_range[o]
                                 for o in model.iter_obj2))

                for o in model.iter_obj2:
                    tmp.append(model.obj_val(o + 1))

                pareto_sols.append(tuple(tmp))

                log += f'solutions: {tmp}'
                if opts.process_logging:
                    logger.info(log)
        else:
            break

    queues.put_result(pareto_sols)


class PyAugmecon(object):

    def __init__(
            self,
            model,
            opts,
            solver_opts={}):

        self.opts = Options(opts, solver_opts)
        self.start_time = time.time()

        self.logs = Logs(self.opts)
        self.logger = logging.getLogger(self.opts.log_name)
        self.logger.setLevel(logging.INFO)
        self.model = Model(model, self.opts)
        self.opts.check(self.model.n_obj)

    def discover_pareto(self):
        self.model.progress.set_message('finding solutions')

        if self.model.min_obj:
            grid_range = list(reversed(range(self.opts.gp)))
        else:
            grid_range = range(self.opts.gp)

        indices = [tuple([n for n in grid_range])
                   for _ in self.model.iter_obj2]
        self.cp = list(itertools.product(*indices))
        self.cp = [i[::-1] for i in self.cp]

        self.model.pickle()
        self.queues = QueueHandler(self.cp, self.opts)
        self.queues.split_work()
        self.procs = ProcessHandler(
            self.opts, solve_grid, self.model, self.queues)

        self.procs.start()
        results = self.queues.get_result(self.procs.procs)
        self.procs.join()
        self.model.clean()

        self.pareto_sols_temp = [i for sublist in results for i in sublist]

    def find_solutions(self):
        def keep_undominated(pts, min):
            pts = np.array(pts)
            undominated = np.ones(pts.shape[0], dtype=bool)
            for i, c in enumerate(pts):
                if undominated[i]:
                    if min:
                        undominated[undominated] = np.any(
                            pts[undominated] < c, axis=1)
                    else:
                        undominated[undominated] = np.any(
                            pts[undominated] > c, axis=1)
                    undominated[i] = True

            return pts[undominated, :]

        # Remove duplicate solutions
        self.sols = list(set(tuple(self.pareto_sols_temp)))
        self.num_sols = len(self.sols)

        # Remove duplicate solutions due to numerical issues by rounding
        self.unique_sols = [tuple(round(sol, self.opts.round) for sol in item)
                            for item in self.sols]
        self.unique_sols = list(set(tuple(self.unique_sols)))
        self.num_unique_sols = len(self.unique_sols)

        # Remove dominated solutions
        self.unique_pareto_sols = keep_undominated(
            self.unique_sols, self.model.min_obj)
        self.num_unique_pareto_sols = len(self.unique_pareto_sols)

    def output_excel(self):
        writer = pd.ExcelWriter(f'{self.logs.logdir}{self.opts.log_name}.xlsx')
        pd.DataFrame(self.model.e).to_excel(writer, 'e_points')
        pd.DataFrame(self.model.payoff).to_excel(writer, 'payoff_table')
        pd.DataFrame(self.sols).to_excel(writer, 'sols')
        pd.DataFrame(self.unique_sols).to_excel(writer, 'unique_sols')
        pd.DataFrame(self.unique_pareto_sols).to_excel(
            writer, 'unique_pareto_sols')
        writer.save()

    def solve(self):
        self.model.construct_payoff()
        self.model.find_obj_range()
        self.model.convert_prob()
        self.discover_pareto()
        self.find_solutions()
        if self.opts.output_excel:
            self.output_excel()

        Helper.clear_line()
        self.runtime = round(time.time() - self.start_time, 2)
        print(f'Solved {self.model.models_solved.value()} models for '
              f'{self.num_unique_pareto_sols} unique Pareto solutions in '
              f'{self.runtime} seconds')

        logger = self.logger
        logger.info(f'Runtime: {self.runtime} seconds')
        logger.info(f'Models solved: {self.model.models_solved.value()}')
        logger.info(f'Infeasibilities: {self.model.infeasibilities.value()}')
        logger.info(f'Solutions: {self.num_sols}')
        logger.info(f'Unique solutions: {self.num_unique_sols}')
        logger.info(f'Unique pareto solutions: {self.num_unique_pareto_sols}')

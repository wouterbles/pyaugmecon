import os
import time
import logging
import itertools
import cloudpickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Value, Lock, Manager
from pathlib import Path
from pyomo.environ import *
from pyomo.opt import TerminationCondition
from pyomo.core.base import (
    Var, Constraint, ConstraintList, maximize, minimize, Set, Param,
    NonNegativeReals, Any)


logging.getLogger('pyomo.core').setLevel(logging.ERROR)

results = []
flag = {}


def collect_result(result):
    global results
    results.append(result)


def clear_line():
    print(' '*80, end='\r', flush=True)


class Counter(object):
    def __init__(self, init_val=0):
        self.val = Value('i', init_val)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


class Progress(object):
    def __init__(self, counter, total, init_message=''):
        self.counter = counter
        self.total = total
        self.message = init_message
        self.bar = ''

    def set_message(self, message):
        self.message = message
        clear_line()

    def increment(self):
        self.counter.increment()
        bar_len = 40

        progress = self.counter.value() / float(self.total)
        filled_len = int(round(bar_len * progress))
        percents = round(100.0 * progress, 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        if self.bar != bar:
            self.bar = bar
            print(
                f'[{bar}] {percents}% ... ({self.message})',
                end='\r',
                flush=True)


def solve_chunk(
        cp,
        obj_minimize,
        objfun_iter2,
        e,
        obj_range,
        g_points,
        flag: dict,
        progress: Progress,
        models_solved: Counter,
        results):

    cp_start = cp[0][0]
    cp_end = cp[g_points - 1][0]
    eps = 1e-3

    model_file = open('model.p', 'rb')
    model = cloudpickle.load(model_file)

    jump = 0
    pareto_sols_temp = []

    for c in cp:
        progress.increment()

        def activate_objfun(model, objfun_index):
            model.obj_list[objfun_index].activate()

        def solve_model(model):
            opt = SolverFactory('gurobi', solver_io='python')
            opt.options['MIPGap'] = 0.0
            opt.options['NonConvex'] = 2
            # opt.options['Threads'] = 1
            return opt.solve(model)

        def do_jump(i, jump):
            return min(jump, abs(cp_end - i))

        def bypass_range(i):
            if i == 0:
                return range(c[i], c[i] + 1)
            elif obj_minimize:
                return range(c[i] - b[i], c[i] + 1)
            else:
                return range(c[i], c[i] + b[i] + 1)

        def early_exit_range(i):
            if i == 0:
                return range(c[i], c[i] + 1)
            elif obj_minimize:
                return range(c[i], cp_start)
            else:
                return range(c[i], cp_end)

        def set_flag_array(flag_range, value, objfun_iter2):
            indices = [tuple([n for n in flag_range(o)])
                       for o in objfun_iter2]
            iter = list(itertools.product(*indices))

            for i in iter:
                flag[i] = value

        if flag.get(c, 0) != 0 and jump == 0:
            jump = do_jump(c[0] - 1, flag[c])

        if jump > 0:
            jump = jump - 1
            continue

        for o in objfun_iter2:
            model.e[o + 2] = e[o, c[o]]

        activate_objfun(model, 1)
        result = solve_model(model)
        models_solved.increment()

        if (result.solver.termination_condition
                != TerminationCondition.optimal):
            set_flag_array(early_exit_range, g_points, objfun_iter2)
            jump = do_jump(c[0], g_points)
            continue
        else:
            b = []

            for i in objfun_iter2:
                step = obj_range[i] / (g_points - 1)
                slack = round(model.Slack[i + 2].value, 3)
                b.append(int(slack/step))

            set_flag_array(bypass_range, b[0] + 1, objfun_iter2)
            jump = do_jump(c[0], b[0])

        # From this point onward the code is about saving and sorting out
        # unique Pareto Optimal Solutions
        temp_list = []

        # If range is to be considered or not, it should also be
        # changed here (otherwise, it produces artifact solutions)
        temp_list.append(round(model.obj_list[1]() - eps
                         * sum(model.Slack[o1].value
                               / obj_range[o1 - 2]
                               for o1 in model.Os), 2))

        for o in objfun_iter2:
            temp_list.append(round(model.obj_list[o + 2](), 2))

        pareto_sols_temp.append(tuple(temp_list))

    results.append(pareto_sols_temp)


class MOOP(object):

    def __init__(
            self,
            base_model,
            options={},
            name='Model name was not defined!'):

        # Define basic process parameters
        self.time_created = time.strftime("%Y%m%d-%H%M%S")
        self.name = name + '_' + str(self.time_created)
        self.model = base_model
        self.start_time = time.time()

        # Configure logging
        logging_folder = 'logs'
        if not os.path.exists(logging_folder):
            os.makedirs(logging_folder)
        self.logdir = f'{Path().absolute()}/{logging_folder}/'
        logfile = f'{self.logdir}{self.name}.log'
        logging.basicConfig(format='%(message)s',
                            filename=logfile, level=logging.INFO)

        # MOOP options
        self.g_points = options.get('grid_points')
        self.nadir_points = options.get('nadir_points')
        self.early_exit = options.get('early_exit', True)
        self.bypass_coefficient = options.get('bypass_coefficient', True)
        self.flag_array = options.get('flag_array', True)
        self.round_decimals = options.get('round_decimals', 2)
        self.eps = options.get('penalty_weight', 1e-3)
        self.nadir_ratio = options.get('nadir_ratio', 1)

        # Solver options
        self.solver_name = options.get('solver_name', 'gurobi')
        self.solver_io = options.get('solver_io', 'python')

        # Logging options
        self.logging_level = options.get('logging_level', 2)
        self.output_excel = options.get('output_excel', False)
        self.output_txt = options.get('output_txt', True)

        self.num_objfun = len(self.model.obj_list)
        self.objfun_iter = range(self.num_objfun)
        self.objfun_iter2 = range(self.num_objfun - 1)
        self.obj_minimize = self.model.obj_list[1].sense == minimize
        models_to_solve = self.g_points**(self.num_objfun - 1) \
            + self.num_objfun*self.num_objfun
        progress_counter = Counter()
        self.progress = Progress(progress_counter, models_to_solve)
        self.models_solved = Counter()
        self.cpu_count = 1
        self.pool = mp.Pool(self.cpu_count)

        if self.g_points is None:
            raise Exception('No number of grid points provided')

        if self.num_objfun < 2:
            raise Exception('Too few objective functions provided')

        if (self.nadir_points is not None and
                len(self.nadir_points) != self.num_objfun - 1):
            raise Exception('Too many or too few nadir points provided')

        self.create_payoff_table()
        self.find_objfun_range()
        self.convert_opt_prob()
        self.discover_pareto()
        self.find_unique_sols()

        clear_line()
        self.runtime = round(time.time() - self.start_time, 2)
        print(f'Solved {self.models_solved.value()} models for '
              f'{self.num_unique_pareto_sols} unique solutions in '
              f'{self.runtime} seconds')

    def round(self, val):
        return round(val, self.round_decimals)

    def activate_objfun(self, objfun_index):
        self.model.obj_list[objfun_index].activate()

    def deactivate_objfun(self, objfun_index):
        self.model.obj_list[objfun_index].deactivate()

    def solve_model(self):
        self.opt = SolverFactory(self.solver_name, solver_io=self.solver_io)
        self.opt.options['MIPGap'] = 0.0
        self.opt.options['NonConvex'] = 2
        # self.opt.options['Threads'] = 1
        self.result = self.opt.solve(self.model)

    def create_payoff_table(self):
        self.progress.set_message('constructing payoff')

        def set_payoff(i, j, is_lexicographic):
            self.activate_objfun(j + 1)
            if is_lexicographic:
                self.model.aux_con = Constraint(
                    expr=self.model.obj_list[i + 1].expr
                    == self.payoff_table[i, i])
            self.solve_model()
            self.progress.increment()
            self.payoff_table[i, j] = self.round(self.model.obj_list[j + 1]())
            self.deactivate_objfun(j + 1)
            if is_lexicographic:
                del self.model.aux_con

        self.payoff_table = np.full(
            (self.num_objfun, self.num_objfun), np.inf)

        # Independently optimize each objective function (diagonal elements)
        for i in self.objfun_iter:
            for j in self.objfun_iter:
                if i == j:
                    set_payoff(i, j, False)

        # Optimize j having all the i as constraints (off-diagonal elements)
        for i in self.objfun_iter:
            for j in self.objfun_iter:
                if i != j:
                    set_payoff(i, j, True)

    def find_objfun_range(self):
        # Gridpoints of p-1 objective functions that are used as constraints
        self.e = np.zeros((self.num_objfun - 1, self.g_points))
        self.obj_range = np.zeros(self.num_objfun - 1)

        for i in self.objfun_iter2:
            if self.nadir_points:
                min = self.nadir_points[i]
            else:
                min = self.nadir_ratio*np.min(self.payoff_table[:, i + 1], 0)

            max = np.max(self.payoff_table[:, i + 1], 0)
            self.obj_range[i] = max - min
            self.e[i] = [min + j*(self.obj_range[i]/(self.g_points - 1))
                         for j in range(0, self.g_points)]

    def convert_opt_prob(self):
        self.model.con_list = ConstraintList()

        # Set of objective functions
        self.model.Os = Set(
            ordered=True,
            initialize=[o + 2 for o in self.objfun_iter2])

        # Slack for objectives introduced as constraints
        self.model.Slack = Var(self.model.Os, within=NonNegativeReals)
        self.model.e = Param(
            self.model.Os,
            initialize=[np.nan for o in self.model.Os],
            within=Any,
            mutable=True)  # RHS of constraints

        # Add p-1 objective functions as constraints
        for o in range(1, self.num_objfun):
            self.model.obj_list[1].expr += self.eps*(
                10**(-1*(o-1))*self.model.Slack[o + 1]
                / self.obj_range[o - 1])

            if self.model.obj_list[o + 1].sense == minimize:
                self.model.con_list.add(
                    expr=self.model.obj_list[o + 1].expr
                    + self.model.Slack[o + 1] == self.model.e[o + 1])
            elif self.model.obj_list[o + 1].sense == maximize:
                self.model.con_list.add(
                    expr=self.model.obj_list[o + 1].expr
                    - self.model.Slack[o + 1] == self.model.e[o + 1])

    def discover_pareto(self):
        if self.obj_minimize:
            grid_range = list(reversed(range(self.g_points)))
        else:
            grid_range = range(self.g_points)

        indices = [tuple([n for n in grid_range])
                   for o in self.objfun_iter2]
        self.cp = list(itertools.product(*indices))
        self.cp = [i[::-1] for i in self.cp]

        # Divide grid points over threads
        self.cp_presplit = [self.cp[i:i + self.g_points]
                            for i in range(0, len(self.cp), self.g_points)]
        remainder = self.g_points % self.cpu_count
        take = int((self.g_points - remainder) / self.cpu_count)
        self.cp_split = []

        start = -take
        for i in range(self.cpu_count):
            start += take
            end = start + take + remainder
            self.cp_split.append([i for sublist in self.cp_presplit[start:end]
                                  for i in sublist])
            if i == 0:
                start += remainder
                remainder = 0

        model_file_name = 'model.p'
        model_file = open(model_file_name, 'wb')
        cloudpickle.dump(self.model, model_file)
        manager = Manager()
        flag = manager.dict()
        results = manager.list()

        self.progress.set_message('finding solutions')
        procs = []  

        for cp in self.cp_split:
            p = mp.Process(
                target=solve_chunk,
                args=(
                    cp,
                    self.obj_minimize,
                    self.objfun_iter2,
                    self.e,
                    self.obj_range,
                    self.g_points,
                    flag,
                    self.progress,
                    self.models_solved,
                    results))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        self.pareto_sols_temp = [i for sublist in results for i in sublist]

    def find_unique_sols(self):
        self.unique_pareto_sols = list(set(tuple(self.pareto_sols_temp)))
        self.num_unique_pareto_sols = len(self.unique_pareto_sols)
        self.pareto_sols = np.zeros(
            (self.num_unique_pareto_sols, self.num_objfun,))

        for item_index, item in enumerate(self.unique_pareto_sols):
            for o in range(self.num_objfun):
                self.pareto_sols[item_index, o] = item[o]

        pd.DataFrame(self.pareto_sols).to_excel(
            f'{self.logdir}{self.name}.xlsx')

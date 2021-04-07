import os
import time
import logging
import datetime
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from pyomo.environ import *
from pyomo.opt import TerminationCondition
from pyomo.core.base import (
    Var, Constraint, ConstraintList, maximize, minimize, Set, Param,
    NonNegativeReals, Any)

logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class MOOP:

    def __init__(
            self,
            base_model,
            options={},
            name='Model name was not defined!'):

        # Define basic process parameters
        self.time_created = datetime.datetime.now()
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
        self.models_to_solve = self.g_points**(self.num_objfun - 1) \
            + self.num_objfun*self.num_objfun
        self.models_solved = 0
        self.progress_count = 0
        self.progress_message = ''

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

        self.clear_line()
        self.runtime = round(time.time() - self.start_time, 2)
        print(f'Solved {self.models_solved} models for '
              f'{self.num_unique_pareto_sols} unique solutions in '
              f'{self.runtime} seconds')

    def round(self, val):
        return round(val, self.round_decimals)

    def clear_line(self):
        print(' '*100, end='\r')

    def progress(self, message):
        bar_len = 50
        self.progress_count += 1
        progress = self.progress_count / float(self.models_to_solve)
        filled_len = int(round(bar_len * progress))

        percents = round(100.0 * progress, 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        if (self.progress_message != message):
            self.clear_line()

        self.progress_message = message
        print(f'[{bar}] {percents}% ... ({message})', end='\r')

    def activate_objfun(self, objfun_index):
        self.model.obj_list[objfun_index].activate()

    def deactivate_objfun(self, objfun_index):
        self.model.obj_list[objfun_index].deactivate()

    def solve_model(self):
        self.opt = SolverFactory(self.solver_name, solver_io=self.solver_io)
        self.opt.options['MIPGap'] = 0.0
        self.opt.options['NonConvex'] = 2
        # self.opt.options['threads'] = 1
        self.result = self.opt.solve(self.model)

    def create_payoff_table(self):
        def set_payoff(i, j, is_lexicographic):
            self.activate_objfun(j + 1)
            if is_lexicographic:
                self.model.aux_con = Constraint(
                    expr=self.model.obj_list[i + 1].expr
                    == self.payoff_table[i, i])
            self.solve_model()
            self.progress('constructing payoff')
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
        self.cp_start = self.cp[0][0]
        self.cp_end = self.cp[self.g_points - 1][0]

        self.flag = {}
        self.jump = 0
        self.pareto_sols_temp = []

        for c in self.cp:
            self.progress('finding solutions')

            def jump(i, jump):
                return min(jump, abs(self.cp_end - i))

            def bypass_range(i):
                if i == 0:
                    return range(c[i], c[i] + 1)
                elif self.obj_minimize:
                    return range(c[i] - b[i], c[i] + 1)
                else:
                    return range(c[i], c[i] + b[i] + 1)

            def early_exit_range(i):
                if i == 0:
                    return range(c[i], c[i] + 1)
                elif self.obj_minimize:
                    return range(c[i], self.cp_start)
                else:
                    return range(c[i], self.cp_end)

            def set_flag_array(flag_range, value):
                if not self.flag_array:
                    return

                indices = [tuple([n for n in flag_range(o)])
                           for o in self.objfun_iter2]
                iter = list(itertools.product(*indices))

                for i in iter:
                    self.flag[i] = value

            if self.flag.get(c, 0) != 0 and self.jump == 0:
                self.jump = jump(c[0] - 1, self.flag[c])

            if self.jump > 0:
                self.jump = self.jump - 1
                continue

            for o in self.objfun_iter2:
                self.model.e[o + 2] = self.e[o, c[o]]
            self.activate_objfun(1)
            self.solve_model()
            self.models_solved += 1

            if (self.result.solver.termination_condition
                    != TerminationCondition.optimal):
                if (self.early_exit):
                    set_flag_array(early_exit_range, self.g_points)
                    self.jump = jump(c[0], self.g_points)

                logging.info(f'{c}, infeasible')
                continue
            elif (self.bypass_coefficient):
                b = []

                for i in self.objfun_iter2:
                    step = self.obj_range[i] / (self.g_points - 1)
                    slack = self.round(self.model.Slack[i + 2].value)
                    b.append(int(slack/step))

                set_flag_array(bypass_range, b[0] + 1)
                self.jump = jump(c[0], b[0])

            # From this point onward the code is about saving and sorting out
            # unique Pareto Optimal Solutions
            temp_list = []

            # If range is to be considered or not, it should also be
            # changed here (otherwise, it produces artifact solutions)
            temp_list.append(self.round(self.model.obj_list[1]() - self.eps
                             * sum(self.model.Slack[o1].value
                                   / self.obj_range[o1 - 2]
                                   for o1 in self.model.Os)))

            for o in self.objfun_iter2:
                temp_list.append(self.round(self.model.obj_list[o + 2]()))

            self.pareto_sols_temp.append(tuple(temp_list))

            logging.info(f'{c}, {temp_list}, {self.jump}')

    def find_unique_sols(self):
        self.unique_pareto_sols = list(set(self.pareto_sols_temp))
        self.num_unique_pareto_sols = len(self.unique_pareto_sols)
        self.pareto_sols = np.zeros(
            (self.num_unique_pareto_sols, self.num_objfun,))

        for item_index, item in enumerate(self.unique_pareto_sols):
            for o in range(self.num_objfun):
                self.pareto_sols[item_index, o] = item[o]

        pd.DataFrame(self.pareto_sols).to_excel(
            f'{self.logdir}{self.name}.xlsx')

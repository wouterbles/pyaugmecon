import numpy
import logging
import datetime
import itertools
import pandas as pd
from pyomo.environ import *
from pyomo.opt import TerminationCondition
from pyomo.core.base import (
    Var, Constraint, ConstraintList, maximize, minimize, Set, Param,
    NonNegativeReals)

logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class MOOP:

    def __init__(
            self,
            base_model,
            moop_options={},
            solver_options={},
            name='Model name was not defined!'):

        # Define basic process parameters
        self.time_created = datetime.datetime.now()
        self.name = name + '_' + str(self.time_created)
        self.model = base_model

        # Configure logging
        logging.basicConfig(filename=f'{self.name}.log', level=logging.INFO)

        # MOOP options
        self.g_points = moop_options.get('grid_points')
        self.nadir_points = moop_options.get('nadir_points')
        self.early_exit = moop_options.get('early_exit')
        self.bypass_coefficient = moop_options.get('bypass_coefficient')
        self.maximize = moop_options.get('maximize')

        # Solver options
        self.solver_name = solver_options.get('solver_name')
        self.solver_io = solver_options.get('solver_io')

        self.num_objfun = len(self.model.obj_list)

        if (self.nadir_points is not None and
                len(self.nadir_points) != self.num_objfun - 1):
            raise Exception('Too many or too few nadir points provided')

        self.create_payoff_table()
        self.find_objfun_range()
        self.convert_opt_prob()
        self.discover_pareto()

    def activate_objfun(self, objfun_index):
        self.model.obj_list[objfun_index].activate()

    def deactivate_objfun(self, objfun_index):
        self.model.obj_list[objfun_index].deactivate()

    def solve_model(self, debug):
        self.opt = SolverFactory(self.solver_name, solver_io=self.solver_io)
        self.opt.options['mipgap'] = 0.0
        self.result = self.opt.solve(self.model)

        if debug:
            print(self.opt.solve(self.model, tee=True))

    def create_payoff_table(self):
        self.payoff_table = numpy.full(
            (self.num_objfun, self.num_objfun), numpy.inf)
        self.ideal_point = numpy.zeros((1, self.num_objfun))

        # Independently optimize each objective function (diagonal elements)
        for i in range(self.num_objfun):
            for j in range(self.num_objfun):  # This defines the active obj fun

                iIn = i + 1
                jIn = j + 1

                if i == j:
                    self.activate_objfun(jIn)
                    self.solve_model(False)
                    self.payoff_table[i, j] = self.model.obj_list[jIn]()
                    self.deactivate_objfun(jIn)
                    self.ideal_point[0, i] = self.model.obj_list[jIn]()

        # Optimize j having all the i as constraints (off-diagonal elements)
        for i in range(self.num_objfun):
            for j in range(self.num_objfun):  # This defines the active obj fun
                iIn = i + 1
                jIn = j + 1

                if i != j:
                    self.activate_objfun(jIn)
                    self.model.aux_con = Constraint(
                        expr=self.model.obj_list[iIn].expr
                        == self.payoff_table[i, i])
                    self.solve_model(False)
                    self.temp_value = self.model.obj_list[jIn]()
                    del self.model.aux_con
                    self.deactivate_objfun(jIn)
                    self.payoff_table[i, j] = round(self.temp_value, 10)

    def find_objfun_range(self):
        # Keeps the gridpoints of p-1 objective functions that are used as
        # constraints
        self.e = numpy.zeros((self.num_objfun - 1, self.g_points))
        # Keeps the range for scaling purposes
        self.obj_range = numpy.array((1, self.num_objfun - 1))

        for i in range(1, self.num_objfun):  # for p-1
            if (self.nadir_points):
                self.min = self.nadir_points[i - 1]
            else:
                self.min = numpy.min(self.payoff_table[:, i], 0)

            self.max = numpy.max(self.payoff_table[:, i], 0)
            self.obj_range[i - 1] = self.max - self.min

            for j in range(0, self.g_points):
                self.e[i - 1, j] = self.min + j * \
                    (self.obj_range[i - 1] / (self.g_points - 1))

    def convert_opt_prob(self):
        self.eps = 10e-3  # Penalty weight in the augmented objective function
        # Set of objective functions
        self.model.Os = Set(
            ordered=True,
            initialize=[o + 1 for o in range(1, self.num_objfun)])

        # Slack for objectives introduced as constraints
        self.model.Slack = Var(self.model.Os, within=NonNegativeReals)
        self.model.e = Param(
            self.model.Os,
            initialize=[
                numpy.nan for o in self.model.Os],
            mutable=True)  # RHS of constraints

        # Modify objective function in case division by objective function
        # range is (un)desirable
        for o in range(self.num_objfun):
            if o != 0:
                self.model.obj_list[1].expr = self.model.obj_list[1].expr \
                    + self.eps*(self.model.Slack[o + 1]/self.obj_range[o - 1])

        print('New objective:', self.model.obj_list[1].expr)

        self.model.con_list = ConstraintList()

        # Add p-1 objective functions as constraints
        for o in range(1, self.num_objfun):
            if self.model.obj_list[o + 1].sense == minimize:
                self.model.con_list.add(
                    expr=self.model.obj_list[o + 1].expr
                    + self.model.Slack[o + 1] == self.model.e[o + 1])

            if self.model.obj_list[o + 1].sense == maximize:
                self.model.con_list.add(
                    expr=self.model.obj_list[o + 1].expr
                    - self.model.Slack[o + 1] == self.model.e[o + 1])

        for o in range(1, self.num_objfun):
            print('Objective as con:', self.model.con_list[o].expr)

    def convert_opt_prob2(self):
        self.eps = 10e-3  # Penalty weight in the augmented objective function
        # Set of objective functions
        self.model.Os = Set(
            ordered=True,
            initialize=[o + 1 for o in range(1, self.num_objfun)])

        # Slack for objectives introduced as constraints
        self.model.Slack = Var(self.model.Os, within=NonNegativeReals)
        self.model.e = Param(
            self.model.Os,
            initialize=[
                numpy.nan for o in self.model.Os],
            mutable=True)  # RHS of constraints

        # Modify objective function in case division by objective function
        # range is (un)desirable
        for o in range(self.num_objfun):
            if o != 0:
                self.model.obj_list[1].expr = self.model.obj_list[1].expr \
                    + self.eps*(
                        10**(-1*(o-1))*self.model.Slack[o + 1]
                        / self.obj_range[o - 1])

        print('New objective:', self.model.obj_list[1].expr)

        self.model.con_list = ConstraintList()

        # Add p-1 objective functions as constraints
        for o in range(1, self.num_objfun):
            if self.model.obj_list[o + 1].sense == minimize:
                self.model.con_list.add(
                    expr=self.model.obj_list[o + 1].expr
                    + self.model.Slack[o + 1] == self.model.e[o + 1])

            if self.model.obj_list[o + 1].sense == maximize:
                self.model.con_list.add(
                    expr=self.model.obj_list[o + 1].expr
                    - self.model.Slack[o + 1] == self.model.e[o + 1])

        for o in range(1, self.num_objfun):
            print('Objective as con:', self.model.con_list[o].expr)

    def discover_pareto(self):
        self.pareto_sols_temp = []

        indices = [tuple([n for n in range(self.g_points)])
                   for o in range(1, self.num_objfun)]

        if not self.maximize:
            indices = [tuple([n for n in reversed(range(self.g_points))])
                       for o in range(1, self.num_objfun)]

        self.cp = list(itertools.product(*indices))
        self.cp = [i[::-1] for i in self.cp]
        self.last_infeasible = None
        self.bypass_jump = 0
        self.models_solved = 0

        for c in self.cp:
            if self.last_infeasible == c[-1]:
                continue

            if self.bypass_jump > 0:
                self.bypass_jump = self.bypass_jump - 1
                continue

            for o in range(1, self.num_objfun):
                self.model.e[o + 1] = self.e[o - 1, c[o - 1]]
            self.activate_objfun(1)
            self.solve_model(False)
            self.models_solved += 1
            # print(c, self.result.solver.termination_condition)

            if (self.early_exit and self.result.solver.termination_condition
                    != TerminationCondition.optimal):
                self.last_infeasible = c[-1]
                self.bypass_jump = 0
                logging.info(f'{c}, infeasible')
                continue

            if (self.bypass_coefficient):
                step = self.obj_range[0] / (self.g_points - 1)
                slack = round(self.model.Slack[2].value, 10)
                jump = int(slack/step)
                until_end = self.g_points - c[0] - 1
                self.bypass_jump = jump if jump < until_end else until_end

            # From this point onward the code is about saving and sorting out
            # unique Pareto Optimal Solutions
            self.temp_list = []

            # If range is to be considered or not, it should also be
            # changed here (otherwise, it produces artifact solutions)
            self.temp_list.append(
                round(self.model.obj_list[1]() - self.eps
                      * sum(
                    self.model.Slack[o1].value / self.obj_range[o1 - 2]
                    for o1 in self.model.Os), 4))

            for o in range(1, self.num_objfun):
                self.temp_list.append(round(self.model.obj_list[o + 1](), 4))

            logging.info(f'{c}, {self.temp_list}, {self.bypass_jump}')

            self.pareto_sols_temp.append(tuple(self.temp_list))

        self.unique_pareto_sols = list(set(self.pareto_sols_temp))
        self.num_unique_pareto_sols = len(self.unique_pareto_sols)
        self.pareto_sols = numpy.zeros(
            (self.num_unique_pareto_sols, self.num_objfun,))

        for item_index, item in enumerate(self.unique_pareto_sols):
            for o in range(self.num_objfun):
                self.pareto_sols[item_index, o] = item[o]

        pd.DataFrame(self.pareto_sols).to_excel(f'{self.name}.xlsx')

import os
import logging
import cloudpickle
import numpy as np
import pyomo.environ as pyo
from pyaugmecon.options import Options
from pyaugmecon.helper import Helper, Counter, ProgressBar
from pyomo.core.base import (
    Var, Constraint, ConstraintList, maximize, minimize, Set, Param,
    NonNegativeReals, Any)


logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class Model(object):
    def __init__(self, model: pyo.ConcreteModel, opts: Options):
        self.model = model
        self.opts = opts

        self.n_obj = len(self.model.obj_list)
        self.iter_obj = range(self.n_obj)
        self.iter_obj2 = range(self.n_obj - 1)
        self.min_obj = self.obj_sense(0) == minimize

        # Setup progress bar
        to_solve = self.opts.gp**(self.n_obj - 1) + self.n_obj**2
        self.progress = ProgressBar(Counter(), to_solve)
        self.models_solved = Counter()

        if self.n_obj < 2:
            raise Exception('Too few objective functions provided')

    def obj(self, i):
        return self.model.obj_list[i + 1]

    def obj_val(self, i):
        return self.obj(i)()

    def obj_expr(self, i):
        return self.obj(i).expr

    def obj_sense(self, i):
        return self.obj(i).sense

    def slack(self, i):
        return self.model.Slack[i + 1]

    def slack_val(self, i):
        return self.slack(i).value

    def obj_activate(self, i):
        self.obj(i).activate()

    def obj_deactivate(self, i):
        self.obj(i).deactivate()

    def solve(self):
        opt = pyo.SolverFactory(
            self.opts.solver_name,
            solver_io=self.opts.solver_io)
        opt.options.update(self.opts.solver_opts)
        self.result = opt.solve(self.model)
        self.term = self.result.solver.termination_condition
        self.status = self.result.solver.status

    def pickle(self):
        model_file = open(self.opts.model_fn, 'wb')
        cloudpickle.dump(self.model, model_file)
        del self.model

    def unpickle(self):
        model_file = open(self.opts.model_fn, 'rb')
        self.model = cloudpickle.load(model_file)

    def clean(self):
        if os.path.exists(self.opts.model_fn):
            os.remove(self.opts.model_fn)

    def is_status_ok(self):
        return self.status == pyo.SolverStatus.ok

    def is_feasible(self):
        return self.term == pyo.TerminationCondition.optimal

    def is_infeasible(self):
        return (self.term == pyo.TerminationCondition.infeasible or
                self.term == pyo.TerminationCondition.infeasibleOrUnbounded)

    def construct_payoff(self):
        self.progress.set_message('constructing payoff')

        def set_payoff(i, j, is_lexicographic):
            self.obj_activate(j)
            if is_lexicographic:
                self.model.aux_con = Constraint(
                    expr=self.obj_expr(i)
                    == self.payoff[i, i])
            self.solve()
            self.progress.increment()
            self.payoff[i, j] = Helper.round(self.obj_val(j))
            self.obj_deactivate(j)
            if is_lexicographic:
                del self.model.aux_con

        self.payoff = np.full((self.n_obj, self.n_obj), np.inf)

        # Independently optimize each objective function (diagonal elements)
        for i in self.iter_obj:
            for j in self.iter_obj:
                if i == j:
                    set_payoff(i, j, False)

        # Optimize j having all the i as constraints (off-diagonal elements)
        for i in self.iter_obj:
            for j in self.iter_obj:
                if i != j:
                    set_payoff(i, j, True)

    def find_obj_range(self):
        # Gridpoints of p-1 objective functions that are used as constraints
        self.e = np.zeros((self.n_obj - 1, self.opts.gp))
        self.obj_range = np.zeros(self.n_obj - 1)

        for i in self.iter_obj2:
            if self.opts.nadir_p:
                min = self.opts.nadir_p[i]
            else:
                min = self.opts.nadir_r*np.min(self.payoff[:, i + 1], 0)

            max = np.max(self.payoff[:, i + 1], 0)
            self.obj_range[i] = max - min
            self.e[i] = [min + j*(self.obj_range[i]/(self.opts.gp - 1))
                         for j in range(0, self.opts.gp)]

    def convert_prob(self):
        self.model.con_list = ConstraintList()

        # Set of objective functions
        self.model.Os = Set(
            ordered=True,
            initialize=[o + 2 for o in self.iter_obj2])

        # Slack for objectives introduced as constraints
        self.model.Slack = Var(self.model.Os, within=NonNegativeReals)
        self.model.e = Param(
            self.model.Os,
            initialize=[np.nan for o in self.model.Os],
            within=Any,
            mutable=True)  # RHS of constraints

        # Add p-1 objective functions as constraints
        for o in range(1, self.n_obj):
            self.model.obj_list[1].expr += self.opts.eps*(
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

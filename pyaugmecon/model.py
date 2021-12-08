import os
import logging
import cloudpickle
import numpy as np
import pyomo.environ as pyo
from pyaugmecon.options import Options
from pyaugmecon.helper import Counter, ProgressBar
from pyomo.core.base import (
    Var,
    ConstraintList,
    maximize,
    minimize,
    Set,
    Param,
    NonNegativeReals,
    Any,
)


class Model(object):
    def __init__(self, model: pyo.ConcreteModel, opts: Options):
        self.model = model
        self.opts = opts
        self.logger = logging.getLogger(opts.log_name)

        self.n_obj = len(self.model.obj_list)
        self.iter_obj = range(self.n_obj)
        self.iter_obj2 = range(self.n_obj - 1)

        # Setup progress bar
        self.to_solve = self.opts.gp ** (self.n_obj - 1) + self.n_obj ** 2
        self.progress = ProgressBar(Counter(), self.to_solve)

        self.models_solved = Counter()
        self.infeasibilities = Counter()

        if self.n_obj < 2:
            raise Exception("Too few objective functions provided")

    def obj(self, i):
        return self.model.obj_list[i + 1]

    def obj_val(self, i):
        return self.obj(i)()

    def obj_expr(self, i):
        return self.obj(i).expr

    def obj_sense(self, i):
        return self.obj(i).sense

    def slack_val(self, i):
        return self.model.Slack[i + 1].value

    def obj_activate(self, i):
        self.obj(i).activate()

    def obj_deactivate(self, i):
        self.obj(i).deactivate()

    def solve(self):
        opt = pyo.SolverFactory(self.opts.solver_name, solver_io=self.opts.solver_io)
        opt.options.update(self.opts.solver_opts)
        self.result = opt.solve(self.model)
        self.term = self.result.solver.termination_condition
        self.status = self.result.solver.status

    def pickle(self):
        model_file = open(self.opts.model_fn, "wb")
        cloudpickle.dump(self.model, model_file)
        del self.model

    def unpickle(self):
        model_file = open(self.opts.model_fn, "rb")
        self.model = cloudpickle.load(model_file)

    def clean(self):
        if os.path.exists(self.opts.model_fn):
            os.remove(self.opts.model_fn)

    def is_optimal(self):
        return (
            self.status == pyo.SolverStatus.ok
            and self.term == pyo.TerminationCondition.optimal
        )

    def is_infeasible(self):
        return (
            self.term == pyo.TerminationCondition.infeasible
            or self.term == pyo.TerminationCondition.infeasibleOrUnbounded
        )

    def min_to_max(self):
        self.obj_goal = [
            -1 if self.obj_sense(o) == minimize else 1 for o in self.iter_obj
        ]

        for o in self.iter_obj:
            if self.obj_sense(o) == minimize:
                self.model.obj_list[o + 1].sense = maximize
                self.model.obj_list[o + 1].expr = -1 * self.model.obj_list[o + 1].expr

    def construct_payoff(self):
        self.logger.info("Constructing payoff")
        self.progress.set_message("constructing payoff")

        def set_payoff(i, j):
            self.obj_activate(j)
            self.solve()
            self.progress.increment()
            self.payoff[i, j] = self.obj_val(j)
            self.obj_deactivate(j)

        self.payoff = np.full((self.n_obj, self.n_obj), np.inf)
        self.model.pcon_list = ConstraintList()

        # Independently optimize each objective function (diagonal elements)
        for i in self.iter_obj:
            for j in self.iter_obj:
                if i == j:
                    set_payoff(i, j)

        # Optimize j having all the i as constraints (off-diagonal elements)
        for i in self.iter_obj:
            self.model.pcon_list.add(expr=self.obj_expr(i) == self.payoff[i, i])

            for j in self.iter_obj:
                if i != j:
                    set_payoff(i, j)
                    self.model.pcon_list.add(expr=self.obj_expr(j) == self.payoff[i, j])

            self.model.pcon_list.clear()

    def find_obj_range(self):
        self.logger.info("Finding objective function range")

        # Gridpoints of p-1 objective functions that are used as constraints
        self.e = np.zeros((self.n_obj - 1, self.opts.gp))
        self.obj_range = np.zeros(self.n_obj - 1)

        for i in self.iter_obj2:
            if self.opts.nadir_p:
                min = self.opts.nadir_p[i]
            else:
                min = self.opts.nadir_r * np.min(self.payoff[:, i + 1], 0)

            max = np.max(self.payoff[:, i + 1], 0)
            self.obj_range[i] = max - min
            self.e[i] = [
                min + j * (self.obj_range[i] / (self.opts.gp - 1))
                for j in range(0, self.opts.gp)
            ]

    def convert_prob(self):
        self.logger.info("Converting optimization problem")

        self.model.con_list = ConstraintList()

        # Set of objective functions
        self.model.Os = Set(ordered=True, initialize=[o + 2 for o in self.iter_obj2])

        # Slack for objectives introduced as constraints
        self.model.Slack = Var(self.model.Os, within=NonNegativeReals)
        self.model.e = Param(
            self.model.Os,
            initialize=[np.nan for _ in self.model.Os],
            within=Any,
            mutable=True,
        )  # RHS of constraints

        # Add p-1 objective functions as constraints
        for o in range(1, self.n_obj):
            self.model.obj_list[1].expr += self.opts.eps * (
                10 ** (-1 * (o - 1)) * self.model.Slack[o + 1] / self.obj_range[o - 1]
            )

            self.model.con_list.add(
                expr=self.model.obj_list[o + 1].expr - self.model.Slack[o + 1]
                == self.model.e[o + 1]
            )

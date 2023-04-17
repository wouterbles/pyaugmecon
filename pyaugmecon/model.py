import logging
import os

import cloudpickle
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.core.base import (
    Any,
    ConstraintList,
    NonNegativeReals,
    Param,
    Set,
    Var,
    maximize,
    minimize,
)

from pyaugmecon.helper import Counter, ProgressBar
from pyaugmecon.options import Options


class Model:
    def __init__(self, model: pyo.ConcreteModel, opts: Options):
        """
        Initialize the Model class.

        Parameters
        ----------
        model : pyo.ConcreteModel
            Pyomo concrete model to solve.
        opts : Options
            Options object for the Pyomo model.

        """
        self.model = model
        self.opts = opts
        self.logger = logging.getLogger(opts.log_name)

        self.n_obj = len(model.obj_list)
        self.iter_obj = range(self.n_obj)
        self.iter_obj2 = range(self.n_obj - 1)

        # Setup progress bar
        self.to_solve = opts.gp ** (self.n_obj - 1) + self.n_obj**2
        self.progress = ProgressBar(Counter(), self.to_solve)

        self.models_solved = Counter()
        self.infeasibilities = Counter()

        if self.n_obj < 2:
            raise ValueError("Too few objective functions provided")

    def obj(self, i):
        """
        Return the i-th objective function.

        Parameters
        ----------
        i : int
            The index of the objective function to return.

        Returns
        -------
        obj : pyo.Objective
            The i-th objective function.

        """
        return self.model.obj_list[i + 1]

    def obj_val(self, i):
        """
        Return the value of the i-th objective function.

        Parameters
        ----------
        i : int
            The index of the objective function to return.

        Returns
        -------
        obj_val : float
            The value of the i-th objective function.

        """
        return self.obj(i)()

    def obj_expr(self, i):
        """
        Return the expression of the i-th objective function.

        Parameters
        ----------
        i : int
            The index of the objective function to return.

        Returns
        -------
        obj_expr : pyo.Expression
            The expression of the i-th objective function.

        """
        return self.obj(i).expr

    def obj_sense(self, i):
        """
        Return the sense of the i-th objective function.

        Parameters
        ----------
        i : int
            The index of the objective function to return.

        Returns
        -------
        obj_sense : pyo.ObjectiveSense
            The sense of the i-th objective function.

        """
        return self.obj(i).sense

    def slack_val(self, i):
        """
        Return the value of the slack variable of the i-th constraint.

        Parameters
        ----------
        i : int
            The index of the constraint.

        Returns
        -------
        slack_val : float
            The value of the slack variable of the i-th constraint.

        """
        return self.model.Slack[i + 1].value

    def obj_activate(self, i):
        """
        Activate the i-th objective function.
        """
        self.obj(i).activate()

    def obj_deactivate(self, i):
        """
        Deactivate the i-th objective function.
        """
        self.obj(i).deactivate()

    def solve(self):
        """
        Solve the model using the specified solver.

        The result, termination condition, and solver status are stored as class attributes.
        """
        opt = pyo.SolverFactory(self.opts.solver_name, solver_io=self.opts.solver_io)
        opt.options.update(self.opts.solver_opts)
        self.result = opt.solve(self.model)
        self.term = self.result.solver.termination_condition
        self.status = self.result.solver.status

    def get_vars(self):
        """
        Return a dictionary containing variable names and values extracted from the Pyomo model.

        Returns
        -------
        vars_dict : dict
            A dictionary containing variable names as keys and Pandas Series containing the variable values as values.
        """
        model_vars = self.model.component_map(ctype=Var, active=True)
        vars_dict = {
            v.name: pd.Series(v.extract_values(), index=v.extract_values().keys()) for v in model_vars.values()
        }
        return vars_dict

    def pickle(self):
        """
        Pickle the Pyomo model to a file.
        """
        model_file = open(self.opts.model_fn, "wb")
        cloudpickle.dump(self.model, model_file)
        del self.model

    def unpickle(self):
        """
        Unpickle the Pyomo model from a file.
        """
        model_file = open(self.opts.model_fn, "rb")
        self.model = cloudpickle.load(model_file)

    def clean(self):
        """
        Remove the Pyomo model file.
        """
        if os.path.exists(self.opts.model_fn):
            os.remove(self.opts.model_fn)

    def is_optimal(self):
        """
        Check if the Pyomo model has been solved optimally.

        Returns
        -------
        bool
            True if the Pyomo model has been solved optimally, False otherwise.
        """
        return self.status == pyo.SolverStatus.ok and self.term == pyo.TerminationCondition.optimal

    def is_infeasible(self):
        """
        Check if the Pyomo model is infeasible.

        Returns
        -------
        bool
            True if the Pyomo model is infeasible, False otherwise.
        """
        return (
            self.term == pyo.TerminationCondition.infeasible
            or self.term == pyo.TerminationCondition.infeasibleOrUnbounded
        )

    def min_to_max(self):
        """
        Convert all minimize objectives to maximize objectives and negate their expressions.

        This method modifies the Pyomo model in-place.

        """
        # Determine objective sense for each objective
        self.obj_goal = [-1 if self.obj_sense(o) == minimize else 1 for o in self.iter_obj]

        # Cconvert minimize objectives to maximize objectives and negate their expressions
        for o in self.iter_obj:
            if self.obj_sense(o) == minimize:
                self.model.obj_list[o + 1].sense = maximize
                self.model.obj_list[o + 1].expr = -self.model.obj_list[o + 1].expr

    def construct_payoff(self):
        """
        Construct a payoff matrix for all pairs of objective functions.

        The payoff matrix is filled with the optimal objective values achieved when optimizing each pair of objectives.

        """
        self.logger.info("Constructing payoff")
        self.progress.set_message("constructing payoff")

        def set_payoff(i, j):
            """
            Helper function that optimizes the Pyomo model with objective function j and saves its value in the payoff
            matrix.

            Parameters
            ----------
            i : int
                The index of the first objective function to use as a constraint.
            j : int
                The index of the second objective function to optimize.

            """
            self.obj_activate(j)
            self.solve()
            self.progress.increment()
            self.payoff[i, j] = self.obj_val(j)
            self.obj_deactivate(j)

        # Initialize payoff matrix with infinity values
        self.payoff = np.full((self.n_obj, self.n_obj), np.inf)
        self.model.pcon_list = ConstraintList()

        # Optimize each objective function independently (diagonal elements)
        for i in self.iter_obj:
            set_payoff(i, i)

        # Optimize each pair of objective functions (off-diagonal elements)
        for i in self.iter_obj:
            self.model.pcon_list.add(expr=self.obj_expr(i) == self.payoff[i, i])

            for j in self.iter_obj:
                if i != j:
                    set_payoff(i, j)
                    self.model.pcon_list.add(expr=self.obj_expr(j) == self.payoff[i, j])

            self.model.pcon_list.clear()

    def find_obj_range(self):
        """
        Find the range of each objective function and create a grid of points that are used as constraints.

        This method modifies the Pyomo model in-place and sets the following class attributes:
            - e : ndarray
                A 2D array containing the gridpoints of p-1 objective functions that are used as constraints.
            - obj_range : ndarray
                A 1D array containing the range of each objective function.

        """
        self.logger.info("Finding objective function range")

        # Initialize gridpoints and objective range arrays
        self.e = np.zeros((self.n_obj - 1, self.opts.gp))
        self.obj_range = np.zeros(self.n_obj - 1)

        # Find range of each objective function
        for i in self.iter_obj2:
            if self.opts.nadir_p:
                obj_min = self.opts.nadir_p[i]
            else:
                obj_min = self.opts.nadir_r * np.min(self.payoff[:, i + 1])

            obj_max = np.max(self.payoff[:, i + 1])
            self.obj_range[i] = obj_max - obj_min
            self.e[i] = np.linspace(obj_min, obj_max, self.opts.gp)

    def convert_prob(self):
        """
        Convert the multi-objective optimization problem to a single-objective optimization problem with constraints.

        This method modifies the Pyomo model in-place and adds the following variables and constraints:
            - Os : Set
                A Pyomo set that contains the indices of the objective functions that are used as constraints.
            - Slack : Var
                A Pyomo variable that represents the slack for each objective function used as a constraint.
            - e : Param
                A Pyomo parameter that contains the gridpoints of the objective functions that are used as constraints.
            - con_list : ConstraintList
                A Pyomo constraint list that contains the constraints for the objective functions.

        """
        self.logger.info("Converting optimization problem")

        # Create constraint list
        self.model.con_list = ConstraintList()

        # Create set of objective functions and Slack variable for each function
        self.model.Os = Set(ordered=True, initialize=[o + 2 for o in self.iter_obj2])
        self.model.Slack = Var(self.model.Os, within=NonNegativeReals)

        # Create parameter for gridpoints of objective functions used as constraints
        self.model.e = Param(self.model.Os, within=Any, mutable=True)

        # Add objective functions as constraints with slack variables
        for o in range(1, self.n_obj):
            self.model.obj_list[1].expr += self.opts.eps * (
                10 ** (-o + 1) * self.model.Slack[o + 1] / self.obj_range[o - 1]
            )

            self.model.con_list.add(
                expr=self.model.obj_list[o + 1].expr - self.model.Slack[o + 1] == self.model.e[o + 1]
            )

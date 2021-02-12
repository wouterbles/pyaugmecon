from numpy.core.numeric import NaN
from pyomo.environ import *
from pyomo.opt import TerminationCondition
import itertools
import numpy
import datetime
import logging
from pyomo.core.base import (
    Var, Constraint, ConstraintList, maximize, minimize, Set, Param,
    NonNegativeReals)

logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class MOOP:

    def __init__(
            self,
            baseModel,
            moopOptions={},
            solverOptions={},
            name='Model name was not defined!'):
        # Define basic process parameters
        self.timeCreated = datetime.datetime.now()
        self.name = name + '_' + str(self.timeCreated)
        self.model = baseModel

        # MOOP options
        self.gPoints = moopOptions['gridPoints']
        self.earlyExit = moopOptions['earlyExit']

        # Solver options
        self.solverName = solverOptions['solverName']
        self.solverIO = solverOptions['solverIO']

        self.numObjFun = len(self.model.objList)

        self.createPayOffTable()
        self.findObjFunRange()
        self.convertOptProb()
        self.discoverPareto()

    def activateObjFun(self, objFunIndex):
        self.model.objList[objFunIndex].activate()

    def deactivateObjFun(self, objFunIndex):
        self.model.objList[objFunIndex].deactivate()

    def solveModel(self):
        self.opt = SolverFactory(self.solverName, solver_io=self.solverIO)
        self.opt.options['mipgap'] = 0.0
        self.result = self.opt.solve(self.model)

    def createPayOffTable(self):
        self.payOffTable = numpy.full(
            (self.numObjFun, self.numObjFun), numpy.inf)
        self.idealPoint = numpy.zeros((1, self.numObjFun))

        # Independently optimize each objective function (diagonal elements)
        for i in range(self.numObjFun):
            for j in range(self.numObjFun):  # This defines the active obj fun

                iIn = i + 1
                jIn = j + 1

                if i == j:
                    self.activateObjFun(jIn)
                    self.solveModel()
                    self.payOffTable[i, j] = self.model.objList[jIn]()
                    self.deactivateObjFun(jIn)
                    self.idealPoint[0, i] = self.model.objList[jIn]()

        # Optimize j having all the i as constraints (off-diagonal elements)
        for i in range(self.numObjFun):
            for j in range(self.numObjFun):  # This defines the active obj fun
                iIn = i + 1
                jIn = j + 1

                if i != j:
                    self.activateObjFun(jIn)
                    self.model.auxCon = Constraint(
                        expr=self.model.objList[iIn].expr
                        == self.payOffTable[i, i])
                    self.solveModel()
                    self.tempValue = self.model.objList[jIn]()
                    del self.model.auxCon
                    self.deactivateObjFun(jIn)
                    self.payOffTable[i, j] = round(self.tempValue, 10)

    def findObjFunRange(self):
        # keeps the gridpoints of p-1 objective functions that are used as
        # constraints
        self.e = numpy.zeros((self.numObjFun - 1, self.gPoints))
        # keeps the range for scaling purposes
        self.objRange = numpy.array((1, self.numObjFun - 1))

        for i in range(1, self.numObjFun):  # for p-1
            self.min = numpy.min(self.payOffTable[:, i], 0)
            self.max = numpy.max(self.payOffTable[:, i], 0)
            self.objRange[i - 1] = self.max - self.min

            for j in range(0, self.gPoints):
                self.e[i - 1, j] = self.min + j * \
                    (self.objRange[i - 1] / (self.gPoints - 1))

    def convertOptProb(self):
        self.eps = 10e-3  # penalty weight in the augmented objective function
        # Set of objective functions
        self.model.Os = Set(
            ordered=True, initialize=[o + 1 for o in range(1, self.numObjFun)])

        # Slack for objectives introduced as constraints
        self.model.Slack = Var(self.model.Os, within=NonNegativeReals)
        self.model.e = Param(
            self.model.Os,
            initialize=[
                numpy.nan for o in self.model.Os],
            mutable=True)  # RHS of constraints

        # Modify objective function in case division by objective function
        # range is (un)desirable
        for o in range(self.numObjFun):
            if o != 0:
                self.model.objList[1].expr = self.model.objList[1].expr + \
                    self.eps * (self.model.Slack[o + 1] / self.objRange[o - 1])

        print('New objective:', self.model.objList[1].expr)

        self.model.objCons = ConstraintList()

        # Add p-1 objective functions as constraints
        for o in range(1, self.numObjFun):
            if self.model.objList[o + 1].sense == minimize:
                self.model.objCons.add(
                    expr=self.model.objList[o + 1].expr
                    + self.model.Slack[o + 1] == self.model.e[o + 1])

            if self.model.objList[o + 1].sense == maximize:
                self.model.objCons.add(
                    expr=self.model.objList[o + 1].expr
                    - self.model.Slack[o + 1] == self.model.e[o + 1])

        for o in range(1, self.numObjFun):
            print('Objective as con:', self.model.objCons[o].expr)

    def discoverPareto(self):
        self.paretoSolsTemp = []

        indices = [tuple([n for n in reversed(range(self.gPoints))])
                   for o in range(1, self.numObjFun)]
        self.cp = list(itertools.product(*indices))
        self.lastInfeasible = None
        self.modelsSolved = 0

        # TODO: this is where early exit should be considered
        for c in self.cp:
            if self.lastInfeasible == c[0]:
                continue

            for o in range(1, self.numObjFun):
                self.model.e[o + 1] = self.e[o - 1, c[o - 1]]
            self.activateObjFun(1)
            self.solveModel()
            self.modelsSolved += 1
            # print(c, self.result.solver.termination_condition)

            if (self.earlyExit and self.result.solver.termination_condition
                    == TerminationCondition.infeasible):
                self.lastInfeasible = c[0]
                continue

            # From this point onward the code is about saving and sorting out
            # unique Pareto Optimal Solutions
            if (self.result.solver.termination_condition
                    != TerminationCondition.infeasible):
                self.tempList = []

                # If range is to be considered or not, it should also be
                # changed here (otherwise, it produces artifact solutions)
                self.tempList.append(
                    self.model.objList[1]() - self.eps
                    * sum(
                        self.model.Slack[o1].value / self.objRange[o1 - 2]
                        for o1 in self.model.Os))

                for o in range(1, self.numObjFun):
                    self.tempList.append(self.model.objList[o + 1]())

                self.paretoSolsTemp.append(tuple(self.tempList))

        self.uniqueParetoSols = list(set(self.paretoSolsTemp))
        self.numUniqueParetoSols = len(self.uniqueParetoSols)
        self.paretoSols = numpy.zeros(
            (self.numUniqueParetoSols, self.numObjFun,))

        for itemIndex, item in enumerate(self.uniqueParetoSols):
            for o in range(self.numObjFun):
                self.paretoSols[itemIndex, o] = item[o]

import numpy as np
from pyaugmecon import *
from tests.optimizationModels import threeObjectiveModel, twoObjectiveModel

MOOPopts = {'gridPoints': 10, 'earlyExit': False}
solverOpts = {'solverName': 'gurobi', 'solverIO': 'python'}

A = MOOP(threeObjectiveModel(), MOOPopts, solverOpts)
print('--- PAY-OFF TABLE ---')
print(A.payOffTable)
print('--')
print(A.idealPoint)
print('--- E-POINTS ---')
print(np.sort(A.e))
print('--')
print(A.objRange)
print('--- PARETO SOLS ---')
print(A.paretoSols.shape)
print('--')
print(A.paretoSols)
print('--- MODELS SOLVED ---')
print(A.modelsSolved)

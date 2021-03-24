import pandas as pd
from pyomo.core.base.set import BinarySet
from pyomo.environ import *
from pyomo.core.base import (
    Var, ConcreteModel, Constraint, ConstraintList, ObjectiveList, maximize,
    minimize, Suffix, Set, Param, NonNegativeReals, Binary)


def two_objective_model():
    model = ConcreteModel()

    # Define variables
    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)

    # --------------------------------------
    #   Define the objective functions
    # --------------------------------------

    def objective1(model):
        return model.x1

    def objective2(model):
        return 3 * model.x1 + 4 * model.x2

    # --------------------------------------
    #   Define the regular constraints
    # --------------------------------------

    def constraint1(model):
        return model.x1 <= 20

    def constraint2(model):
        return model.x2 <= 40

    def constraint3(model):
        return 5 * model.x1 + 4 * model.x2 <= 200

    # --------------------------------------
    #   Add components to the model
    # --------------------------------------

    # Add the constraints to the model
    model.con1 = Constraint(rule=constraint1)
    model.con2 = Constraint(rule=constraint2)
    model.con3 = Constraint(rule=constraint3)

    # Add the objective functions to the model using ObjectiveList(). Note
    # that the first index is 1 instead of 0!
    model.obj_list = ObjectiveList()
    model.obj_list.add(expr=objective1(model), sense=maximize)
    model.obj_list.add(expr=objective2(model), sense=maximize)

    # By default deactivate all the objective functions
    for o in range(len(model.obj_list)):
        model.obj_list[o + 1].deactivate()

    return model


def three_objective_model():
    model = ConcreteModel()

    # Define variables
    model.LIGN = Var(within=NonNegativeReals)
    model.LIGN1 = Var(within=NonNegativeReals)
    model.LIGN2 = Var(within=NonNegativeReals)
    model.OIL = Var(within=NonNegativeReals)
    model.OIL2 = Var(within=NonNegativeReals)
    model.OIL3 = Var(within=NonNegativeReals)
    model.NG = Var(within=NonNegativeReals)
    model.NG1 = Var(within=NonNegativeReals)
    model.NG2 = Var(within=NonNegativeReals)
    model.NG3 = Var(within=NonNegativeReals)
    model.RES = Var(within=NonNegativeReals)
    model.RES1 = Var(within=NonNegativeReals)
    model.RES3 = Var(within=NonNegativeReals)

    # --------------------------------------
    #   Define the objective functions
    # --------------------------------------

    def objective1(model):
        return 30*model.LIGN + 75*model.OIL + 60*model.NG + 90*model.RES

    def objective2(model):
        return 1.44*model.LIGN + 0.72*model.OIL + 0.45*model.NG

    def objective3(model):
        return model.OIL + model.NG

    # --------------------------------------
    #   Define the regular constraints
    # --------------------------------------

    def constraint1(model):
        return model.LIGN - model.LIGN1 - model.LIGN2 == 0

    def constraint2(model):
        return model.OIL - model.OIL2 - model.OIL3 == 0

    def constraint3(model):
        return model.NG - model.NG1 - model.NG2 - model.NG3 == 0

    def constraint4(model):
        return model.RES - model.RES1 - model.RES3 == 0

    def constraint5(model):
        return model.LIGN <= 31000

    def constraint6(model):
        return model.OIL <= 15000

    def constraint7(model):
        return model.NG <= 22000

    def constraint8(model):
        return model.RES <= 10000

    def constraint9(model):
        return model.LIGN1 + model.NG1 + model.RES1 >= 38400

    def constraint10(model):
        return model.LIGN2 + model.OIL2 + model.NG2 >= 19200

    def constraint11(model):
        return model.OIL3 + model.NG3 + model.RES3 >= 6400

    # --------------------------------------
    #   Add components to the model
    # --------------------------------------

    # Add the constraints to the model
    model.con1 = Constraint(rule=constraint1)
    model.con2 = Constraint(rule=constraint2)
    model.con3 = Constraint(rule=constraint3)
    model.con4 = Constraint(rule=constraint4)
    model.con5 = Constraint(rule=constraint5)
    model.con6 = Constraint(rule=constraint6)
    model.con7 = Constraint(rule=constraint7)
    model.con8 = Constraint(rule=constraint8)
    model.con9 = Constraint(rule=constraint9)
    model.con10 = Constraint(rule=constraint10)
    model.con11 = Constraint(rule=constraint11)

    # Add the objective functions to the model using ObjectiveList(). Note
    # that the first index is 1 instead of 0!
    model.obj_list = ObjectiveList()
    model.obj_list.add(expr=objective1(model), sense=minimize)
    model.obj_list.add(expr=objective2(model), sense=minimize)
    model.obj_list.add(expr=objective3(model), sense=minimize)

    # By default deactivate all the objective functions
    for o in range(len(model.obj_list)):
        model.obj_list[o + 1].deactivate()

    return model


def two_kp_model(type):
    model = ConcreteModel()

    # Define input files
    xlsx = pd.ExcelFile(f"tests/input/{type}.xlsx")
    a = pd.read_excel(xlsx, index_col=0, sheet_name='a').to_numpy()
    b = pd.read_excel(xlsx, index_col=0, sheet_name='b').to_numpy()
    c = pd.read_excel(xlsx, index_col=0, sheet_name='c').to_numpy()

    # Define variables
    model.ITEMS = Set(initialize=range(len(a[0])))
    model.x = Var(model.ITEMS, within=Binary)

    # --------------------------------------
    #   Define the objective functions
    # --------------------------------------

    def objective1(model):
        return sum(c[0][i]*model.x[i] for i in model.ITEMS)

    def objective2(model):
        return sum(c[1][i]*model.x[i] for i in model.ITEMS)

    # --------------------------------------
    #   Define the regular constraints
    # --------------------------------------

    def constraint1(model):
        return sum(a[0][i]*model.x[i] for i in model.ITEMS) <= b[0][0]

    def constraint2(model):
        return sum(a[1][i]*model.x[i] for i in model.ITEMS) <= b[1][0]

    # --------------------------------------
    #   Add components to the model
    # --------------------------------------

    # Add the constraints to the model
    model.con1 = Constraint(rule=constraint1)
    model.con2 = Constraint(rule=constraint2)

    # Add the objective functions to the model using ObjectiveList(). Note
    # that the first index is 1 instead of 0!
    model.obj_list = ObjectiveList()
    model.obj_list.add(expr=objective1(model), sense=maximize)
    model.obj_list.add(expr=objective2(model), sense=maximize)

    # By default deactivate all the objective functions
    for o in range(len(model.obj_list)):
        model.obj_list[o + 1].deactivate()

    return model


def three_kp_model(type):
    model = ConcreteModel()

    # Define input files
    xlsx = pd.ExcelFile(f"tests/input/{type}.xlsx")
    a = pd.read_excel(xlsx, index_col=0, sheet_name='a').to_numpy()
    b = pd.read_excel(xlsx, index_col=0, sheet_name='b').to_numpy()
    c = pd.read_excel(xlsx, index_col=0, sheet_name='c').to_numpy()

    # Define variables
    model.ITEMS = Set(initialize=range(len(a[0])))
    model.x = Var(model.ITEMS, within=Binary)

    # --------------------------------------
    #   Define the objective functions
    # --------------------------------------

    def objective1(model):
        return sum(c[0][i]*model.x[i] for i in model.ITEMS)

    def objective2(model):
        return sum(c[1][i]*model.x[i] for i in model.ITEMS)

    def objective3(model):
        return sum(c[2][i]*model.x[i] for i in model.ITEMS)

    # --------------------------------------
    #   Define the regular constraints
    # --------------------------------------

    def constraint1(model):
        return sum(a[0][i]*model.x[i] for i in model.ITEMS) <= b[0][0]

    def constraint2(model):
        return sum(a[1][i]*model.x[i] for i in model.ITEMS) <= b[1][0]

    def constraint3(model):
        return sum(a[2][i]*model.x[i] for i in model.ITEMS) <= b[2][0]

    # --------------------------------------
    #   Add components to the model
    # --------------------------------------

    # Add the constraints to the model
    model.con1 = Constraint(rule=constraint1)
    model.con2 = Constraint(rule=constraint2)
    model.con3 = Constraint(rule=constraint3)

    # Add the objective functions to the model using ObjectiveList(). Note
    # that the first index is 1 instead of 0!
    model.obj_list = ObjectiveList()
    model.obj_list.add(expr=objective1(model), sense=maximize)
    model.obj_list.add(expr=objective2(model), sense=maximize)
    model.obj_list.add(expr=objective3(model), sense=maximize)

    # By default deactivate all the objective functions
    for o in range(len(model.obj_list)):
        model.obj_list[o + 1].deactivate()

    return model


def four_kp_model(type):
    model = ConcreteModel()

    # Define input files
    xlsx = pd.ExcelFile(f"tests/input/{type}.xlsx")
    a = pd.read_excel(xlsx, index_col=0, sheet_name='a').to_numpy()
    b = pd.read_excel(xlsx, index_col=0, sheet_name='b').to_numpy()
    c = pd.read_excel(xlsx, index_col=0, sheet_name='c').to_numpy()

    # Define variables
    model.ITEMS = Set(initialize=range(len(a[0])))
    model.x = Var(model.ITEMS, within=Binary)

    # --------------------------------------
    #   Define the objective functions
    # --------------------------------------

    def objective1(model):
        return sum(c[0][i]*model.x[i] for i in model.ITEMS)

    def objective2(model):
        return sum(c[1][i]*model.x[i] for i in model.ITEMS)

    def objective3(model):
        return sum(c[2][i]*model.x[i] for i in model.ITEMS)

    def objective4(model):
        return sum(c[3][i]*model.x[i] for i in model.ITEMS)

    # --------------------------------------
    #   Define the regular constraints
    # --------------------------------------

    def constraint1(model):
        return sum(a[0][i]*model.x[i] for i in model.ITEMS) <= b[0][0]

    def constraint2(model):
        return sum(a[1][i]*model.x[i] for i in model.ITEMS) <= b[1][0]

    def constraint3(model):
        return sum(a[2][i]*model.x[i] for i in model.ITEMS) <= b[2][0]

    def constraint4(model):
        return sum(a[3][i]*model.x[i] for i in model.ITEMS) <= b[3][0]

    # --------------------------------------
    #   Add components to the model
    # --------------------------------------

    # Add the constraints to the model
    model.con1 = Constraint(rule=constraint1)
    model.con2 = Constraint(rule=constraint2)
    model.con3 = Constraint(rule=constraint3)
    model.con4 = Constraint(rule=constraint4)

    # Add the objective functions to the model using ObjectiveList(). Note
    # that the first index is 1 instead of 0!
    model.obj_list = ObjectiveList()
    model.obj_list.add(expr=objective1(model), sense=maximize)
    model.obj_list.add(expr=objective2(model), sense=maximize)
    model.obj_list.add(expr=objective3(model), sense=maximize)
    model.obj_list.add(expr=objective4(model), sense=maximize)

    # By default deactivate all the objective functions
    for o in range(len(model.obj_list)):
        model.obj_list[o + 1].deactivate()

    return model
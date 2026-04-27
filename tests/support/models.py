from pyomo.core.base import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    ObjectiveList,
    Var,
    maximize,
    minimize,
)


def two_objective_model():
    model = ConcreteModel()

    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)

    model.con1 = Constraint(expr=model.x1 <= 20)
    model.con2 = Constraint(expr=model.x2 <= 40)
    model.con3 = Constraint(expr=5 * model.x1 + 4 * model.x2 <= 200)

    model.obj_list = ObjectiveList()
    model.obj_list.add(expr=model.x1, sense=maximize)
    model.obj_list.add(expr=3 * model.x1 + 4 * model.x2, sense=maximize)

    return model


def three_objective_model():
    model = _make_energy_model()

    model.obj_list.add(expr=model.cost_expr, sense=minimize)
    model.obj_list.add(expr=model.emissions_expr, sense=minimize)
    model.obj_list.add(expr=model.fuel_expr, sense=minimize)

    return model


def three_objective_mixed_model():
    model = _make_energy_model()

    model.obj_list.add(expr=model.cost_expr, sense=minimize)
    model.obj_list.add(expr=-model.emissions_expr, sense=maximize)
    model.obj_list.add(expr=model.fuel_expr, sense=minimize)

    return model


def _make_energy_model():
    model = ConcreteModel()

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

    model.con1 = Constraint(expr=model.LIGN - model.LIGN1 - model.LIGN2 == 0)
    model.con2 = Constraint(expr=model.OIL - model.OIL2 - model.OIL3 == 0)
    model.con3 = Constraint(expr=model.NG - model.NG1 - model.NG2 - model.NG3 == 0)
    model.con4 = Constraint(expr=model.RES - model.RES1 - model.RES3 == 0)
    model.con5 = Constraint(expr=model.LIGN <= 31000)
    model.con6 = Constraint(expr=model.OIL <= 15000)
    model.con7 = Constraint(expr=model.NG <= 22000)
    model.con8 = Constraint(expr=model.RES <= 10000)
    model.con9 = Constraint(expr=model.LIGN1 + model.NG1 + model.RES1 >= 38400)
    model.con10 = Constraint(expr=model.LIGN2 + model.OIL2 + model.NG2 >= 19200)
    model.con11 = Constraint(expr=model.OIL3 + model.NG3 + model.RES3 >= 6400)

    model.cost_expr = 30 * model.LIGN + 75 * model.OIL + 60 * model.NG + 90 * model.RES
    model.emissions_expr = 1.44 * model.LIGN + 0.72 * model.OIL + 0.45 * model.NG
    model.fuel_expr = model.OIL + model.NG

    model.obj_list = ObjectiveList()

    return model

import pandas as pd
from pathlib import Path
from tests.helper import Helper
from pyomo.core.base import (
    Var,
    ConcreteModel,
    Constraint,
    ObjectiveList,
    maximize,
    minimize,
    Set,
    Param,
    NonNegativeReals,
    Binary,
)


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
        return 30 * model.LIGN + 75 * model.OIL + 60 * model.NG + 90 * model.RES

    def objective2(model):
        return 1.44 * model.LIGN + 0.72 * model.OIL + 0.45 * model.NG

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


def three_objective_mixed_model():
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
        return 30 * model.LIGN + 75 * model.OIL + 60 * model.NG + 90 * model.RES

    def objective2(model):
        return -1 * (1.44 * model.LIGN + 0.72 * model.OIL + 0.45 * model.NG)

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
    model.obj_list.add(expr=objective2(model), sense=maximize)
    model.obj_list.add(expr=objective3(model), sense=minimize)

    # By default deactivate all the objective functions
    for o in range(len(model.obj_list)):
        model.obj_list[o + 1].deactivate()

    return model


def unit_commitment_model():
    model = ConcreteModel()

    # Define input files
    xlsx = pd.ExcelFile(
        f"{Path(__file__).parent.absolute()}/input/unit_commitment.xlsx",
        engine="openpyxl",
    )
    system_demand = Helper.read_excel(xlsx, "SystemDemand")
    storage_systems = Helper.read_excel(xlsx, "StorageSystems")
    generators = Helper.read_excel(xlsx, "Generators")
    generator_step_size = Helper.read_excel(xlsx, "GeneratorStepSize")
    generator_step_cost = Helper.read_excel(xlsx, "GeneratorStepCost")
    pv_generation = Helper.read_excel(xlsx, "PVGeneration")

    # Define sets
    model.T = Set(ordered=True, initialize=system_demand.index)
    model.I = Set(ordered=True, initialize=generators.index)
    model.F = Set(ordered=True, initialize=generator_step_size.columns)
    model.S = Set(ordered=True, initialize=storage_systems.index)

    # Define parameters
    model.Pmax = Param(model.I, within=NonNegativeReals, mutable=True)
    model.Pmin = Param(model.I, within=NonNegativeReals, mutable=True)

    model.RU = Param(model.I, within=NonNegativeReals, mutable=True)
    model.RD = Param(model.I, within=NonNegativeReals, mutable=True)
    model.SUC = Param(model.I, within=NonNegativeReals, mutable=True)
    model.SDC = Param(model.I, within=NonNegativeReals, mutable=True)
    model.Pini = Param(model.I, within=NonNegativeReals, mutable=True)
    model.uini = Param(model.I, within=Binary, mutable=True)
    model.C = Param(model.I, model.F, within=NonNegativeReals, mutable=True)
    model.B = Param(model.I, model.F, within=NonNegativeReals, mutable=True)
    model.SystemDemand = Param(model.T, within=NonNegativeReals, mutable=True)
    model.Emissions = Param(model.I, within=NonNegativeReals, mutable=True)

    model.PV = Param(model.T, within=NonNegativeReals, mutable=True)

    model.ESS_Pmax = Param(model.S, within=NonNegativeReals, mutable=True)
    model.ESS_SOEmax = Param(model.S, within=NonNegativeReals, mutable=True)
    model.ESS_SOEini = Param(model.S, within=NonNegativeReals, mutable=True)
    model.ESS_Eff = Param(model.S, within=NonNegativeReals, mutable=True)

    # Give values to parameters of the generators
    for i in model.I:
        model.Pmin[i] = generators.loc[i, "Pmin"]
        model.Pmax[i] = generators.loc[i, "Pmax"]
        model.RU[i] = generators.loc[i, "RU"]
        model.RD[i] = generators.loc[i, "RD"]
        model.SUC[i] = generators.loc[i, "SUC"]
        model.SDC[i] = generators.loc[i, "SDC"]
        model.Pini[i] = generators.loc[i, "Pini"]
        model.uini[i] = generators.loc[i, "uini"]
        model.Emissions[i] = generators.loc[i, "Emissions"]
        for f in model.F:
            model.B[i, f] = generator_step_size.loc[i, f]
            model.C[i, f] = generator_step_cost.loc[i, f]

    # Add system demand and PV generation
    for t in model.T:
        model.SystemDemand[t] = system_demand.loc[t, "SystemDemand"]
        model.PV[t] = pv_generation.loc[t, "PVGeneration"]

    # Give values to ESS parameters
    for s in model.S:
        model.ESS_Pmax[s] = storage_systems.loc[s, "Power"]
        model.ESS_SOEmax[s] = storage_systems.loc[s, "Energy"]
        model.ESS_SOEini[s] = storage_systems.loc[s, "SOEini"]
        model.ESS_Eff[s] = storage_systems.loc[s, "Eff"]

    # Define decision variables
    model.P = Var(model.I, model.T, within=NonNegativeReals)
    model.Pres = Var(model.T, within=NonNegativeReals)
    model.b = Var(model.I, model.F, model.T, within=NonNegativeReals)
    model.u = Var(model.I, model.T, within=Binary)
    model.CSU = Var(model.I, model.T, within=NonNegativeReals)
    model.CSD = Var(model.I, model.T, within=NonNegativeReals)

    model.SOE = Var(model.S, model.T, within=NonNegativeReals)
    model.Pch = Var(model.S, model.T, within=NonNegativeReals)
    model.Pdis = Var(model.S, model.T, within=NonNegativeReals)
    model.u_ess = Var(model.S, model.T, within=Binary)

    # --------------------------------------
    #   Define the objective functions
    # --------------------------------------

    def cost_objective(model):
        return sum(
            sum(
                sum(model.C[i, f] * model.b[i, f, t] for f in model.F)
                + model.CSU[i, t]
                + model.CSD[i, t]
                for i in model.I
            )
            for t in model.T
        )

    def emissions_objective(model):
        return sum(
            sum(model.P[i, t] * model.Emissions[i] for i in model.I) for t in model.T
        )

    def unmet_objective(model):
        return sum(model.Pres[t] for t in model.T)

    # --------------------------------------
    #   Define the regular constraints
    # --------------------------------------

    def power_decomposition_rule1(model, i, t):
        return model.P[i, t] == sum(model.b[i, f, t] for f in model.F)

    def power_decomposition_rule2(model, i, f, t):
        return model.b[i, f, t] <= model.B[i, f]

    def power_min_rule(model, i, t):
        return model.P[i, t] >= model.Pmin[i] * model.u[i, t]

    def power_max_rule(model, i, t):
        return model.P[i, t] <= model.Pmax[i] * model.u[i, t]

    def ramp_up_rule(model, i, t):
        if model.T.ord(t) == 1:
            return model.P[i, t] - model.Pini[i] <= 60 * model.RU[i]

        if model.T.ord(t) > 1:
            return model.P[i, t] - model.P[i, model.T.prev(t)] <= 60 * model.RU[i]

    def ramp_down_rule(model, i, t):
        if model.T.ord(t) == 1:
            return (model.Pini[i] - model.P[i, t]) <= 60 * model.RD[i]

        if model.T.ord(t) > 1:
            return (model.P[i, model.T.prev(t)] - model.P[i, t]) <= 60 * model.RD[i]

    def start_up_cost(model, i, t):
        if model.T.ord(t) == 1:
            return model.CSU[i, t] >= model.SUC[i] * (model.u[i, t] - model.uini[i])

        if model.T.ord(t) > 1:
            return model.CSU[i, t] >= model.SUC[i] * (
                model.u[i, t] - model.u[i, model.T.prev(t)]
            )

    def shut_down_cost(model, i, t):
        if model.T.ord(t) == 1:
            return model.CSD[i, t] >= model.SDC[i] * (model.uini[i] - model.u[i, t])

        if model.T.ord(t) > 1:
            return model.CSD[i, t] >= model.SDC[i] * (
                model.u[i, model.T.prev(t)] - model.u[i, t]
            )

    def ESS_SOEupdate(model, s, t):
        if model.T.ord(t) == 1:
            return (
                model.SOE[s, t]
                == model.ESS_SOEini[s]
                + model.ESS_Eff[s] * model.Pch[s, t]
                - model.Pdis[s, t] / model.ESS_Eff[s]
            )

        if model.T.ord(t) > 1:
            return (
                model.SOE[s, t]
                == model.SOE[s, model.T.prev(t)]
                + model.ESS_Eff[s] * model.Pch[s, t]
                - model.Pdis[s, t] / model.ESS_Eff[s]
            )

    def ESS_SOElimit(model, s, t):
        return model.SOE[s, t] <= model.ESS_SOEmax[s]

    def ESS_Charging(model, s, t):
        return model.Pch[s, t] <= model.ESS_Pmax[s] * model.u_ess[s, t]

    def ESS_Discharging(model, s, t):
        return model.Pdis[s, t] <= model.ESS_Pmax[s] * (1 - model.u_ess[s, t])

    def Balance(model, t):
        return model.PV[t] + sum(model.P[i, t] for i in model.I) + sum(
            model.Pdis[s, t] for s in model.S
        ) == model.SystemDemand[t] - model.Pres[t] + sum(
            model.Pch[s, t] for s in model.S
        )

    def Pres_max(model, t):
        return model.Pres[t] <= 0.1 * model.SystemDemand[t]

    # --------------------------------------
    #   Add components to the model
    # --------------------------------------

    # Add the constraints to the model
    model.power_decomposition_rule1 = Constraint(
        model.I, model.T, rule=power_decomposition_rule1
    )
    model.power_decomposition_rule2 = Constraint(
        model.I, model.F, model.T, rule=power_decomposition_rule2
    )
    model.power_min_rule = Constraint(model.I, model.T, rule=power_min_rule)
    model.power_max_rule = Constraint(model.I, model.T, rule=power_max_rule)
    model.start_up_cost = Constraint(model.I, model.T, rule=start_up_cost)
    model.shut_down_cost = Constraint(model.I, model.T, rule=shut_down_cost)
    model.ConSOEUpdate = Constraint(model.S, model.T, rule=ESS_SOEupdate)
    model.ConCharging = Constraint(model.S, model.T, rule=ESS_Charging)
    model.ConDischarging = Constraint(model.S, model.T, rule=ESS_Discharging)
    model.ConSOElimit = Constraint(model.S, model.T, rule=ESS_SOElimit)
    model.ConGenUp = Constraint(model.I, model.T, rule=ramp_up_rule)
    model.ConGenDown = Constraint(model.I, model.T, rule=ramp_down_rule)
    model.ConBalance = Constraint(model.T, rule=Balance)
    model.Pres_max = Constraint(model.T, rule=Pres_max)

    # Add the objective functions to the model using ObjectiveList(). Note
    # that the first index is 1 instead of 0!
    model.obj_list = ObjectiveList()
    model.obj_list.add(expr=cost_objective(model), sense=minimize)
    model.obj_list.add(expr=emissions_objective(model), sense=minimize)
    model.obj_list.add(expr=unmet_objective(model), sense=minimize)

    # By default deactivate all the objective functions
    for o in range(len(model.obj_list)):
        model.obj_list[o + 1].deactivate()

    return model


def two_kp_model(type):
    model = ConcreteModel()

    # Define input files
    xlsx = pd.ExcelFile(
        f"{Path(__file__).parent.absolute()}/input/{type}.xlsx", engine="openpyxl"
    )
    a = pd.read_excel(xlsx, index_col=0, sheet_name="a").to_numpy()
    b = pd.read_excel(xlsx, index_col=0, sheet_name="b").to_numpy()
    c = pd.read_excel(xlsx, index_col=0, sheet_name="c").to_numpy()

    # Define variables
    model.ITEMS = Set(initialize=range(len(a[0])))
    model.x = Var(model.ITEMS, within=Binary)

    # --------------------------------------
    #   Define the objective functions
    # --------------------------------------

    def objective1(model):
        return sum(c[0][i] * model.x[i] for i in model.ITEMS)

    def objective2(model):
        return sum(c[1][i] * model.x[i] for i in model.ITEMS)

    # --------------------------------------
    #   Define the regular constraints
    # --------------------------------------

    def constraint1(model):
        return sum(a[0][i] * model.x[i] for i in model.ITEMS) <= b[0][0]

    def constraint2(model):
        return sum(a[1][i] * model.x[i] for i in model.ITEMS) <= b[1][0]

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
    xlsx = pd.ExcelFile(
        f"{Path(__file__).parent.absolute()}/input/{type}.xlsx", engine="openpyxl"
    )
    a = pd.read_excel(xlsx, index_col=0, sheet_name="a").to_numpy()
    b = pd.read_excel(xlsx, index_col=0, sheet_name="b").to_numpy()
    c = pd.read_excel(xlsx, index_col=0, sheet_name="c").to_numpy()

    # Define variables
    model.ITEMS = Set(initialize=range(len(a[0])), within=NonNegativeReals)
    model.x = Var(model.ITEMS, within=Binary)

    # --------------------------------------
    #   Define the objective functions
    # --------------------------------------

    def objective1(model):
        return sum(c[0][i] * model.x[i] for i in model.ITEMS)

    def objective2(model):
        return sum(c[1][i] * model.x[i] for i in model.ITEMS)

    def objective3(model):
        return sum(c[2][i] * model.x[i] for i in model.ITEMS)

    # --------------------------------------
    #   Define the regular constraints
    # --------------------------------------

    def constraint1(model):
        return sum(a[0][i] * model.x[i] for i in model.ITEMS) <= b[0][0]

    def constraint2(model):
        return sum(a[1][i] * model.x[i] for i in model.ITEMS) <= b[1][0]

    def constraint3(model):
        return sum(a[2][i] * model.x[i] for i in model.ITEMS) <= b[2][0]

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
    xlsx = pd.ExcelFile(
        f"{Path(__file__).parent.absolute()}/input/{type}.xlsx", engine="openpyxl"
    )
    a = pd.read_excel(xlsx, index_col=0, sheet_name="a").to_numpy()
    b = pd.read_excel(xlsx, index_col=0, sheet_name="b").to_numpy()
    c = pd.read_excel(xlsx, index_col=0, sheet_name="c").to_numpy()

    # Define variables
    model.ITEMS = Set(initialize=range(len(a[0])))
    model.x = Var(model.ITEMS, within=Binary)

    # --------------------------------------
    #   Define the objective functions
    # --------------------------------------

    def objective1(model):
        return sum(c[0][i] * model.x[i] for i in model.ITEMS)

    def objective2(model):
        return sum(c[1][i] * model.x[i] for i in model.ITEMS)

    def objective3(model):
        return sum(c[2][i] * model.x[i] for i in model.ITEMS)

    def objective4(model):
        return sum(c[3][i] * model.x[i] for i in model.ITEMS)

    # --------------------------------------
    #   Define the regular constraints
    # --------------------------------------

    def constraint1(model):
        return sum(a[0][i] * model.x[i] for i in model.ITEMS) <= b[0][0]

    def constraint2(model):
        return sum(a[1][i] * model.x[i] for i in model.ITEMS) <= b[1][0]

    def constraint3(model):
        return sum(a[2][i] * model.x[i] for i in model.ITEMS) <= b[2][0]

    def constraint4(model):
        return sum(a[3][i] * model.x[i] for i in model.ITEMS) <= b[3][0]

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

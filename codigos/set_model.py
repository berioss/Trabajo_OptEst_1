import gurobipy as gp
from gurobipy import GRB

def set_model(n_nodes: int, N: list[int]):
    """
    Construye el maestro de 1ª etapa (VRP/TSP con MTZ).
    - N: lista de clientes (sin depósito). Se asume depósito=0.
    Devuelve: (model, x, w), donde x[(i,j)] son binarias y w[i] son MTZ.
    """
    depot = 0
    cardN = len(N)
    m = gp.Model("master_first_stage")
    m.Params.OutputFlag = 0

    # --- Vars ---
    x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
         for i in range(n_nodes) for j in range(n_nodes) if i != j}

    # MTZ (solo clientes)
    w = {i: m.addVar(lb=1.0, ub=float(cardN), vtype=GRB.CONTINUOUS, name=f"w_{i}")
         for i in N}

    m.update()

    # --- Grado 1 en clientes ---
    # salida única de cada cliente
    for i in N:
        m.addConstr(gp.quicksum(x[i, j] for j in range(n_nodes) if j != i) == 1,
                    name=f"out_{i}")
    # entrada única a cada cliente
    for j in N:
        m.addConstr(gp.quicksum(x[i, j] for i in range(n_nodes) if i != j) == 1,
                    name=f"in_{j}")

    # --- Depósito: una salida y una entrada ---
    m.addConstr(gp.quicksum(x[depot, j] for j in range(1, n_nodes)) == 1,
                name="depot_out")
    m.addConstr(gp.quicksum(x[i, depot] for i in range(1, n_nodes)) == 1,
                name="depot_in")

    # --- MTZ para eliminar subtours entre clientes ---
    for i in N:
        for j in N:
            if i == j:
                continue
            m.addConstr(w[i] - w[j] + cardN * x[i, j] <= cardN - 1,
                        name=f"mtz_{i}_{j}")

    m.update()
    return m, x, w


def set_objective_first_stage(model: gp.Model, x: dict, c: dict[tuple[int, int], float]):
    """
    Fija la FO:  min sum_{i!=j} c_{ij} x_{ij}.
    c: dict {(i,j): costo} para todos los arcos i!=j.
    """
    obj = gp.quicksum(c[(i, j)] * x[(i, j)] for (i, j) in x.keys())
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()


def extract_route(x_sol: dict[tuple[int, int], int | float], depot: int = 0) -> list[int]:
    """
    Reconstruye el ciclo (incluye regreso al depósito) a partir de x_{ij} binaria.
    """
    nxt = {}
    for (i, j), v in x_sol.items():
        if v > 0.5:
            nxt[i] = j
    route = [depot]
    cur = depot
    seen = {depot}
    while True:
        cur = nxt[cur]
        route.append(cur)
        if cur == depot:
            break
        if cur in seen:  # por seguridad
            break
        seen.add(cur)
    return route


def solve_model(n_nodes: int, N: list[int], c: dict[tuple[int, int], float],
                 timelimit: float | None = None, mipgap: float | None = None,
                 verbose: bool = False):
    """
    Construye, fija FO de 1ª etapa y resuelve. Devuelve dict con resultados.
    """
    m, x, w = set_model(n_nodes, N)
    set_objective_first_stage(m, x, c)

    if timelimit is not None:
        m.Params.TimeLimit = timelimit
    if mipgap is not None:
        m.Params.MIPGap = mipgap
    m.Params.OutputFlag = 1 if verbose else 0

    m.optimize()

    x_sol = {k: int(round(var.X)) for k, var in x.items()}
    route = extract_route(x_sol, depot=0)
    return {
        "obj": m.ObjVal,
        "x": x_sol,
        "route": route,
        "status": m.Status
    }

import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

def build_c(n_nodes,df_s):
    n = n_nodes
    c = {(i,j): df_s[(df_s["i"] == i) & (df_s["j"] == j)]['t_min'].values[0] for i in range(n) for j in range(n) if i!=j}
    return c

def solve_lshaped(
    V,                  # lista de nodos (ej. [0,1,...,11])
    depot,              # índice del depósito
    c,                  # dict {(i,j): costo determinista}
    T_list,             # lista de escenarios: dict {(i,j): tiempo}
    H,                  # horizonte (ej. 480)
    d,                  # dict {i: demanda} (solo i in N)
    alpha,              # dict {i: productividad}
    c_out=2.0, c_OT=1.0,
    gap_tol=1e-6, max_iters=50,
    verbose=False,
    diagnostic=False     # imprime Q, lam, tau, RHS, theta por escenario
):
    """
    L-shaped entero con theta por escenario y cortes de optimalidad.
    Mantiene MTZ en el maestro. Mismos argumentos que tu versión original.

    Retorna:
        {
          "obj": valor_objetivo_final,
          "x_sol": {(i,j): 0/1},
          "route": [ruta desde 'depot' cerrando en 'depot'],
          "history": [{"iter":..,"master_obj":..,"cuts_added_iter":..,"cuts_total":..}, ...],
          "cuts_added": total_cortes,
          "master_model": modelo_gurobi
        }
    """

    # --- Conjuntos y arcos
    N = [i for i in V if i != depot]
    arcs = [(i, j) for i in V for j in V if i != j]
    K = len(T_list)

    # --- Maestro
    def build_master():
        m = gp.Model("master")
        m.Params.OutputFlag = 1 if verbose else 0

        x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
        u = m.addVars(N, lb=1, ub=len(N), vtype=GRB.CONTINUOUS, name="u")
        theta = {s: m.addVar(lb=0.0, name=f"theta_{s}") for s in range(K)}

        # Grado 1
        for i in V:
            m.addConstr(gp.quicksum(x[i, j] for j in V if j != i) == 1, name=f"out_{i}")
            m.addConstr(gp.quicksum(x[j, i] for j in V if j != i) == 1, name=f"in_{i}")

        # MTZ
        for i in N:
            for j in N:
                if i == j:
                    continue
                m.addConstr(u[i] - u[j] + len(N) * x[i, j] <= len(N) - 1, name=f"mtz_{i}_{j}")

        # Objetivo
        det_cost = gp.quicksum(c[(i,j)] * x[i, j] for (i, j) in arcs)
        m.setObjective(det_cost + (1.0 / K) * gp.quicksum(theta[s] for s in range(K)), GRB.MINIMIZE)
        return m, x, u, theta

    # --- Subproblema (primal) por escenario, dado x_fix
    def build_primal(T_s, x_fix):
        """
        Variables: u_i >=0, r_i >=0, o >=0
        Restricciones:
            u_i + r_i = d_i
            sum(u_i/alpha_i) + tau(x_fix) <= H + o
        Objetivo: min c_out * sum r_i + c_OT * o
        Devuelve: (modelo optimizado, refs_constr, tau)
        """
        sp = gp.Model("sub")
        sp.Params.OutputFlag = 0

        # τ(x) robusto a llaves faltantes en T_s
        tau = sum(T_s.get((i, j), 0.0) * x_fix.get((i, j), 0) for i in V for j in V if i != j)

        u_var = sp.addVars(N, lb=0.0, name="u")
        r_var = sp.addVars(N, lb=0.0, name="r")
        o_var = sp.addVar(lb=0.0, name="o")

        demand_constr = {}
        for i in N:
            demand_constr[i] = sp.addConstr(u_var[i] + r_var[i] == d[i], name=f"demand_{i}")

        time_constr = sp.addConstr(
            gp.quicksum(u_var[i] / alpha[i] for i in N) + tau <= H + o_var,
            name="time_cap"
        )

        sp.setObjective(c_out * gp.quicksum(r_var[i] for i in N) + c_OT * o_var, GRB.MINIMIZE)
        sp.optimize()

        if sp.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Subproblema no óptimo (status={sp.Status})")

        refs = {"demand_constr": demand_constr, "time_constr": time_constr}
        return sp, refs, tau

    # --- Utilidad: extraer ruta de x binaria
    def get_route(x_sol):
        nxt = {i: j for (i, j), val in x_sol.items() if val > 0.5}
        route = [depot]
        seen = set()
        while True:
            cur = route[-1]
            if cur in seen:  # protección ante ciclos raros
                break
            seen.add(cur)
            j = nxt.get(cur, depot)
            route.append(j)
            if j == depot:
                break
        return route

    # =========================
    #       BUCLE L-SHAPED
    # =========================
    m_master, x_vars, _, theta_vars = build_master()
    cuts_added_total = 0
    history = []

    for it in range(1, max_iters + 1):
        if verbose:
            print(f"\n--- Iteración {it} ---")

        m_master.optimize()
        if m_master.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
            raise RuntimeError(f"Maestro falló (status={m_master.Status})")

        x_fix = {(i, j): int(round(x_vars[i, j].X)) for (i, j) in arcs}
        master_obj = m_master.ObjVal

        cuts_added_iter = 0

        # Por escenario: resolver sub, extraer duales, chequear violación y añadir corte si aplica
        for s_idx, T_s in enumerate(T_list):
            sp, cref, tau = build_primal(T_s=T_s, x_fix=x_fix)

            # Duales
            pi  = {i: cref["demand_constr"][i].Pi for i in N}   # libres
            lam = - cref["time_constr"].Pi                      # CORRECCIÓN DE SIGNO: λ ≥ 0

            # RHS del corte: pi·d + lam*(H - tau)
            pi_di   = sum(pi[i] * d[i] for i in N)
            rhs     = pi_di + lam * (H - tau)
            theta_v = theta_vars[s_idx].X

            if diagnostic:
                print(f"[it {it:02d} esc {s_idx:03d}] Q={sp.ObjVal:.3f}  lam={lam:.6f}  tau={tau:.2f}  RHS={rhs:.6f}  theta={theta_v:.6f}")

            if rhs - theta_v > gap_tol:
                # theta_s >= (pi·d + lam H) + sum (-lam * T_ij) x_ij
                const_term = pi_di + lam * H
                expr = const_term + gp.quicksum(
                    (-lam) * T_s.get((i, j), 0.0) * x_vars[i, j] for i in V for j in V if i != j
                )
                m_master.addConstr(theta_vars[s_idx] >= expr, name=f"optcut_s{s_idx}_it{it}")
                cuts_added_iter += 1
                cuts_added_total += 1

                if verbose:
                    print(f"  Esc {s_idx}: corte agregado | RHS={rhs:.6f} > theta={theta_v:.6f} (lam={lam:.6f})")

        history.append({
            "iter": it,
            "master_obj": master_obj,
            "cuts_added_iter": cuts_added_iter,
            "cuts_total": cuts_added_total
        })

        if verbose:
            print(f"Iter {it}: cortes añadidos = {cuts_added_iter}")

        # Parada por no violación
        if cuts_added_iter == 0:
            if verbose:
                print("Convergencia: no se añadieron cortes en esta iteración.")
            break

    # Optimiza una vez más si quedó con cortes nuevos
    if m_master.Status != GRB.OPTIMAL:
        m_master.optimize()

    x_sol = {(i, j): int(round(x_vars[i, j].X)) for (i, j) in arcs}
    route = get_route(x_sol)
    final_obj = m_master.ObjVal

    return {
        "obj": final_obj,
        "x_sol": x_sol,
        "route": route,
        "history": history,
        "cuts_added": cuts_added_total,
        "master_model": m_master,
    }


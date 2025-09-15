import numpy as np
import pandas as pd

def sample_SAA(pools: dict, K: int = 50, seed: int | None = 123):
    """
    Devuelve:
      - T_list: lista de K diccionarios, cada uno con tiempos {(i,j): t_ij}
      - T_frames: lista de K DataFrames (i,j,t)
    """
    rng = np.random.default_rng(seed)
    arcs = sorted(pools.keys())
    T_list, T_frames = [], []

    for s in range(K):
        Ts = {}
        rows = []
        for arc in arcs:
            pool = pools[arc]
            if len(pool) == 0:
                Ts[arc] = np.nan  # o decide una política (p.ej., saltar el arco)
            else:
                Ts[arc] = rng.choice(pool, replace=True)
            rows.append((arc[0], arc[1], Ts[arc]))
        T_list.append(Ts)
        T_frames.append(pd.DataFrame(rows, columns=["i","j","t_min"]))
    return T_list, T_frames



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def boxplot_por_arco(df: pd.DataFrame, *, figsize=(14,6), ordenar=True, guardar=None):
    """
    df: DataFrame con columnas ['i','j','t_min'] (una fila = un viaje del pool).
    """
    data = df.copy()
    data["arco"] = data["i"].astype(str) + "→" + data["j"].astype(str)

    if ordenar:
        # orden por i y luego j
        orden = (data
                 .drop_duplicates(["i","j"])
                 .sort_values(["i","j"])
                 .assign(arco=lambda d: d["i"].astype(str) + "→" + d["j"].astype(str))
                 ["arco"].tolist())
    else:
        orden = None

    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x="arco", y="t_min", order=orden)
    plt.xticks(rotation=90)
    plt.xlabel("Arco i→j")
    plt.ylabel("Duración (min)")
    plt.title("Distribución de duraciones por arco (boxplot)")
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=200)
    plt.show()


def pools_to_df(pools: dict[tuple[int, int], list[float]]) -> pd.DataFrame:
    rows = []
    for (i, j), tiempos in pools.items():
        rows.extend([(i, j, t) for t in tiempos])
    return pd.DataFrame(rows, columns=["i", "j", "t_min"])


def clean_outliers_iqr(pools: dict[tuple[int, int], list[float]],
                       min_size: int = 50) -> dict:
    """
    Elimina outliers de cada lista usando el criterio IQR.
    Si un pool queda con menos de `min_size` datos, se deja como estaba.
    """
    cleaned = {}
    for arc, tiempos in pools.items():
        x = np.array(tiempos, dtype=float)
        if x.size < min_size:
            cleaned[arc] = x.tolist()
            continue

        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        filtrado = x[(x >= lower) & (x <= upper)]
        if filtrado.size >= min_size:
            cleaned[arc] = filtrado.tolist()
        else:
            cleaned[arc] = x.tolist()  # si se quedó con muy pocos, no limpiar

    return cleaned

def build_T_list(pools: dict, K: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    T_list = []

    arcs = sorted(pools.keys())
    for _ in range(K):
        T_s = {}
        for arc in arcs:
            pool = pools[arc]
            if len(pool) == 0:
                raise ValueError(f"Pool vacío para arco {arc}")
            T_s[arc] = rng.choice(pool, replace=True)
        T_list.append(T_s)
    return T_list

def dataframe_to_c(df):
    """
    Convierte un DataFrame cuadrado de distancias a un dict {(i,j): d_ij},
    excluyendo la diagonal.
    """
    c = {}
    for i in df.index:
        for j in df.columns:
            if i != j:
                c[(i, j)] = float(df.loc[i, j])
    return c

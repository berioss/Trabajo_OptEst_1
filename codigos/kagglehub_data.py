import os
import kagglehub
import pandas as pd
from glob import glob
from pathlib import Path

def get_data() -> pd.DataFrame:
    """
    Descarga y filtra Yellow Taxi (enero 2015) con:
      - Lunes a viernes
      - 09:00–17:00 (pickup y dropoff)
      - Coordenadas no nulas
      - Duración t en [1, 120] minutos
    Guarda/lee cache en 'data.csv'.
    """
    cache = Path("csv/data.csv")

    if cache.exists():
        df = pd.read_csv(
            cache,
            parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        )
        print("Data loaded from file.")
        return df

    # 1) Descargar dataset (última versión del repo de Kaggle)
    path = kagglehub.dataset_download("elemento/nyc-yellow-taxi-trip-data")

    # 2) Localizar el CSV de enero 2015 (nombre típico: yellow_tripdata_2015-01.csv)
    candidates = glob(os.path.join(path, "*2015-01*.csv"))
    if not candidates:
        raise FileNotFoundError("No se encontró el archivo de enero 2015 en el dataset descargado.")
    file = candidates[0]

    # 3) Columnas que necesitamos (esquema 2015)
    cols = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude"
    ]

    # 4) Cargar solo columnas necesarias
    df = pd.read_csv(
        file,
        usecols=cols,
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    )

    # 5) Filtrar días hábiles (0=lunes,...,4=viernes)
    df = df[
        (df.tpep_pickup_datetime.dt.weekday < 5) &
        (df.tpep_dropoff_datetime.dt.weekday < 5)
    ]

    # 6) Filtrar franja 09:00–17:00 (pickup y dropoff)
    t9  = pd.to_datetime("09:00:00").time()
    t17 = pd.to_datetime("17:00:00").time()
    df = df[
        (df.tpep_pickup_datetime.dt.time >= t9)  &
        (df.tpep_pickup_datetime.dt.time <= t17) &
        (df.tpep_dropoff_datetime.dt.time  >= t9) &
        (df.tpep_dropoff_datetime.dt.time  <= t17)
    ]

    # 7) Coordenadas válidas (descartar nulos/0)
    for c in ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]:
        df = df[df[c].notna()]
    df = df[
        (df["pickup_longitude"]  != 0) & (df["pickup_latitude"]  != 0) &
        (df["dropoff_longitude"] != 0) & (df["dropoff_latitude"] != 0)
    ]

    # 8) Duración del viaje y limpieza de outliers (1–120 min)
    dur_min = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    df = df[(dur_min >= 1.0) & (dur_min <= 120.0)].copy()
    df["duration_min"] = dur_min.loc[df.index]

    # 9) Guardar cache
    df.to_csv(cache, index=False)
    print("Data saved and loaded.")
    return df

import numpy as np
import pandas as pd


import numpy as np
import pandas as pd

# ---------- Distancia euclidiana aprox. en metros ----------
def euclidean_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """
    Distancia euclidiana aproximada en metros entre arrays (lat1, lon1) y el punto fijo (lat2, lon2).
    Usa factores m/° con ajuste longitudinal por cos(lat2).
    """
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat2))

    dlat = (lat1 - lat2) * m_per_deg_lat
    dlon = (lon1 - lon2) * m_per_deg_lon
    return np.sqrt(dlat**2 + dlon**2)

# ---------- Preprocesamiento con filtros y duración ----------
def preprocess_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Quita coordenadas faltantes o 0
    - Calcula duración (min) y filtra en [1, 120]
    Devuelve un DataFrame limpio con 'duration_min'.
    """
    cols_coords = ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]
    df = df.dropna(subset=cols_coords).copy()

    # coords != 0
    mask_coords = (
        (df["pickup_longitude"]  != 0) &
        (df["pickup_latitude"]   != 0) &
        (df["dropoff_longitude"] != 0) &
        (df["dropoff_latitude"]  != 0)
    )
    df = df.loc[mask_coords].copy()

    # duración
    dur_min = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    df = df.loc[(dur_min >= 1.0) & (dur_min <= 120.0)].copy()
    df.loc[:, "duration_min"] = dur_min.loc[df.index].to_numpy()
    return df

# ---------- Pools O-D con radios crecientes y fallback horario ----------
def build_pools(
    df: pd.DataFrame,
    nodes: dict,
    r_init: int = 250,
    r_max: int = 400,
    step: int = 50,
    m_min: int = 50,
    allow_time_fallback: bool = True
) -> tuple[dict, pd.DataFrame]:
    """
    Crea pools P_ij de duraciones:
      1) L-V, 09:00–17:00 (supuesto base de la práctica)
      2) Para cada (i,j), aumenta r: 250,300,350,400 m hasta lograr >= m_min
      3) Si sigue < m_min y allow_time_fallback=True, amplía a 08:00–18:00 y reintenta
    Devuelve: (pools, summary_df)
    """
    # asegurarnos de estar en días hábiles:
    df = df.loc[
        (df.tpep_pickup_datetime.dt.weekday < 5) &
        (df.tpep_dropoff_datetime.dt.weekday < 5)
    ].copy()

    # ventanas horarias
    t9,  t17  = pd.to_datetime("09:00:00").time(), pd.to_datetime("17:00:00").time()
    t8,  t18  = pd.to_datetime("08:00:00").time(), pd.to_datetime("18:00:00").time()

    def filter_by_hours(data, t_start, t_end):
        return data.loc[
            (data.tpep_pickup_datetime.dt.time >= t_start) &
            (data.tpep_pickup_datetime.dt.time <= t_end) &
            (data.tpep_dropoff_datetime.dt.time >= t_start) &
            (data.tpep_dropoff_datetime.dt.time <= t_end)
        ]

    # split por horario (para reusar rápido)
    df_narrow = filter_by_hours(df, t9, t17)
    df_wide   = filter_by_hours(df, t8, t18)

    # arrays (evitamos overhead de .values muchas veces)
    def arrays_from(data):
        return (
            data["pickup_latitude"].to_numpy(),
            data["pickup_longitude"].to_numpy(),
            data["dropoff_latitude"].to_numpy(),
            data["dropoff_longitude"].to_numpy(),
            data["duration_min"].to_numpy()
        )

    pick_lat_n, pick_lon_n, drop_lat_n, drop_lon_n, dur_n = arrays_from(df_narrow)
    pick_lat_w, pick_lon_w, drop_lat_w, drop_lon_w, dur_w = arrays_from(df_wide)

    # pre-cómputo de distancias a cada nodo para ambos horarios (acelera el while)
    # nodes: dict[i] = (name, lat, lon)
    nodes_list = sorted(nodes.items())  # [(i, (name, lat, lon)), ...]
    I = [i for i, _ in nodes_list]

    # dist pick/drop a cada nodo, horario "narrow"
    Dpick_n = {}
    Ddrop_n = {}
    Dpick_w = {}
    Ddrop_w = {}

    for i, (_, lat_i, lon_i) in nodes_list:
        Dpick_n[i] = euclidean_distance(pick_lat_n, pick_lon_n, lat_i, lon_i)
        Ddrop_n[i] = euclidean_distance(drop_lat_n, drop_lon_n, lat_i, lon_i)
        Dpick_w[i] = euclidean_distance(pick_lat_w, pick_lon_w, lat_i, lon_i)
        Ddrop_w[i] = euclidean_distance(drop_lat_w, drop_lon_w, lat_i, lon_i)

    pools = {}
    summary = []

    for i, _ in nodes_list:
        for j, _ in nodes_list:
            if i == j:
                continue

            # 1) intentar con 09–17
            r = r_init
            pool = []
            used_window = "09-17"
            while r <= r_max and len(pool) < m_min:
                mask = (Dpick_n[i] < r) & (Ddrop_n[j] < r)
                pool = dur_n[mask].tolist()
                if len(pool) < m_min:
                    r += step

            # 2) fallback horario si sigue corto
            final_radius = r if len(pool) >= m_min else None
            if (len(pool) < m_min) and allow_time_fallback:
                r2 = r_init
                pool2 = []
                while r2 <= r_max and len(pool2) < m_min:
                    mask2 = (Dpick_w[i] < r2) & (Ddrop_w[j] < r2)
                    pool2 = dur_w[mask2].tolist()
                    if len(pool2) < m_min:
                        r2 += step
                if len(pool2) >= m_min:
                    pool = pool2
                    final_radius = r2
                    used_window = "08-18"

            pools[(i, j)] = pool
            summary.append({
                "i": i,
                "j": j,
                "n_obs": len(pool),
                "final_radius_m": final_radius,
                "time_window": used_window if len(pool) >= m_min else "insuficiente"
            })
            print(f"Arc ({i}->{j}): {len(pool)} obs, ventana={used_window}, radio_final={final_radius}")

    summary_df = pd.DataFrame(summary, columns=["i","j","n_obs","final_radius_m","time_window"])
    return pools, summary_df

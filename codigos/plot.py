import matplotlib.pyplot as plt

def plot_route_basic(coords: dict[int, tuple[float,float]],
                     route: list[int],
                     depot: int = 0,
                     figsize=(7, 9)):
    """
    Dibuja la ruta como líneas rectas entre puntos (lat, lon).
    - coords: {i: (lat, lon)}
    - route: lista como [0, 7, 1, 8, ..., 0] (incluye el regreso al depósito)
    """
    # Extrae puntos en el orden de la ruta
    lats = [coords[i][0] for i in route]
    lons = [coords[i][1] for i in route]

    fig, ax = plt.subplots(figsize=figsize)
    # Conecta con líneas
    ax.plot(lons, lats, '-', linewidth=2, alpha=0.9)
    # Dibuja puntos
    ax.scatter(lons, lats, s=60, zorder=3)

    # Etiquetas: índice en la ruta
    for k, i in enumerate(route):
        ax.text(coords[i][1], coords[i][0], f"{k}",
                fontsize=9, ha="left", va="bottom")

    # Resalta depósito
    dep_lat, dep_lon = coords[depot]
    ax.scatter([dep_lon], [dep_lat], s=120, marker='*', zorder=4)
    ax.text(dep_lon, dep_lat, "DEP", fontsize=10, weight="bold",
            ha="right", va="top")

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Ruta óptima (orden de visita)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Versión con mapa base (opcional) ----------------------------------------
def plot_route_with_basemap(coords: dict[int, tuple[float,float]],
                            route: list[int],
                            depot: int = 0,
                            figsize=(7, 9)):
    """
    Igual que la básica, pero proyecta a Web Mercator y agrega mapa base.
    Requiere: geopandas, shapely, pyproj, contextily
    """
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    import contextily as cx

    # GeoDataFrame de puntos
    pts = gpd.GeoDataFrame(
        {"node": route, "order": list(range(len(route)))},
        geometry=[Point(coords[i][1], coords[i][0]) for i in route],  # (lon, lat)
        crs="EPSG:4326"
    )
    # Línea de la ruta
    line = gpd.GeoDataFrame(geometry=[LineString(pts.geometry.tolist())], crs="EPSG:4326")

    # Proyectar a Web Mercator (ms tiles)
    pts_3857  = pts.to_crs(3857)
    line_3857 = line.to_crs(3857)

    fig, ax = plt.subplots(figsize=figsize)
    line_3857.plot(ax=ax, color="tab:blue", linewidth=2, alpha=0.9)
    pts_3857.plot(ax=ax, color="black", markersize=25, zorder=3)

    # Etiquetas orden de visita
    for (_, row) in pts_3857.iterrows():
        ax.text(row.geometry.x, row.geometry.y, str(row["order"]),
                fontsize=9, ha="left", va="bottom")

    # Depósito
    dep_pt = gpd.GeoSeries([Point(coords[depot][1], coords[depot][0])], crs="EPSG:4326").to_crs(3857)
    dep_pt.plot(ax=ax, color="gold", markersize=80, marker="*", zorder=4)
    ax.text(dep_pt.iloc[0].x, dep_pt.iloc[0].y, "DEP", fontsize=10, weight="bold",
            ha="right", va="top")

    # Mapa base
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    ax.set_axis_off()
    plt.title("Ruta óptima sobre mapa base")
    plt.tight_layout()
    plt.show()
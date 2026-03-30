from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import random
import sys

import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import LineString

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.common.project_paths import CITY_CACHE_DIR, CITY_RESULTS_DIR, REAL_CITY_FIGURES_DIR, ensure_pipeline_directories


TXDOT_AADT_LAYER = "https://services.arcgis.com/KTcxiTD9dsQw4r7Z/ArcGIS/rest/services/TxDOT_AADT/FeatureServer/0/query"
TXDOT_AADT_FIELD = "AADT_CUR"

CITY_CONFIGS = {
    "austin": {
        "place": "Austin, Texas, USA",
        "center": (30.2672, -97.7431),
        "radius_m": 5500,
    },
    "houston": {
        "place": "Houston, Texas, USA",
        "center": (29.7604, -95.3698),
        "radius_m": 6500,
    },
    "dallas": {
        "place": "Dallas, Texas, USA",
        "center": (32.7767, -96.7970),
        "radius_m": 6000,
    },
}

ATTACK_LABELS = {
    "phantom_congestion": "Phantom congestion",
    "false_closure": "False closure",
    "signal_spoofing": "Signal spoofing",
}


@dataclass
class RouteSample:
    origin: int
    destination: int
    path: List[int]
    weight: float


def ensure_directories(root: Path) -> Dict[str, Path]:
    paths = {
        "cache": CITY_CACHE_DIR,
        "figures": REAL_CITY_FIGURES_DIR,
        "generated": CITY_RESULTS_DIR,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _bounds_to_envelope(bounds: Tuple[float, float, float, float]) -> str:
    west, south, east, north = bounds
    return f"{west},{south},{east},{north}"


def download_txdot_aadt(bounds: Tuple[float, float, float, float], cache_path: Path) -> gpd.GeoDataFrame:
    if cache_path.exists():
        return gpd.read_file(cache_path)

    params = {
        "where": "AADT_CUR IS NOT NULL",
        "geometry": _bounds_to_envelope(bounds),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "AADT_CUR,RTE_NM,RTE_PRFX,RTE_NBR,SYSTEM",
        "returnGeometry": "true",
        "f": "geojson",
    }
    response = requests.get(TXDOT_AADT_LAYER, params=params, timeout=120)
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    return gpd.read_file(cache_path)


def download_city_graph(place: str, center: Tuple[float, float], radius_m: int, cache_path: Path) -> nx.MultiDiGraph:
    if cache_path.exists():
        return ox.load_graphml(cache_path)

    graph = ox.graph_from_point(center, dist=radius_m, network_type="drive", simplify=True)
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    ox.save_graphml(graph, cache_path)
    return graph


def annotate_edges_with_aadt(graph: nx.MultiDiGraph, aadt_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
    edges_gdf = edges_gdf.reset_index()
    edges_gdf = edges_gdf.to_crs(3857)
    aadt_lines = aadt_gdf[~aadt_gdf.geometry.is_empty].copy()
    aadt_lines = aadt_lines.to_crs(3857)

    if aadt_lines.empty:
        edges_gdf["AADT_CUR"] = pd.NA
        return edges_gdf

    joined = gpd.sjoin_nearest(
        edges_gdf,
        aadt_lines[[TXDOT_AADT_FIELD, "geometry"]],
        how="left",
        distance_col="aadt_distance_m",
    )
    joined["AADT_CUR"] = pd.to_numeric(joined[TXDOT_AADT_FIELD], errors="coerce")
    return joined


def build_edge_weight_lookup(edges_gdf: gpd.GeoDataFrame) -> Dict[Tuple[int, int, int], float]:
    lookup: Dict[Tuple[int, int, int], float] = {}
    for row in edges_gdf.itertuples():
        weight = getattr(row, "AADT_CUR", None)
        if pd.isna(weight) or weight is None:
            weight = 1.0
        lookup[(row.u, row.v, row.key)] = float(weight)
    return lookup


def select_weighted_nodes(graph: nx.MultiDiGraph, edges_gdf: gpd.GeoDataFrame) -> List[Tuple[int, float]]:
    node_weights: Dict[int, float] = {node: 1.0 for node in graph.nodes}
    for row in edges_gdf.itertuples():
        aadt = getattr(row, "AADT_CUR", None)
        if pd.isna(aadt) or aadt is None:
            continue
        node_weights[row.u] = node_weights.get(row.u, 1.0) + float(aadt)
        node_weights[row.v] = node_weights.get(row.v, 1.0) + float(aadt)
    return list(node_weights.items())


def weighted_choice(rng: random.Random, weighted_items: Sequence[Tuple[int, float]]) -> int:
    nodes = [item[0] for item in weighted_items]
    weights = [item[1] for item in weighted_items]
    return rng.choices(nodes, weights=weights, k=1)[0]


def sample_routes(
    graph: nx.MultiDiGraph,
    weighted_nodes: Sequence[Tuple[int, float]],
    count: int,
    seed: int,
) -> List[RouteSample]:
    rng = random.Random(seed)
    samples: List[RouteSample] = []

    for _ in range(count * 5):
        if len(samples) >= count:
            break
        origin = weighted_choice(rng, weighted_nodes)
        destination = weighted_choice(rng, weighted_nodes)
        if origin == destination:
            continue
        try:
            path = nx.shortest_path(graph, origin, destination, weight="travel_time")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) < 4:
            continue
        samples.append(RouteSample(origin=origin, destination=destination, path=path, weight=1.0))

    return samples


def route_lines(graph: nx.MultiDiGraph, routes: Iterable[RouteSample]) -> gpd.GeoDataFrame:
    geometries = []
    for route in routes:
        coords = []
        for node_id in route.path:
            node = graph.nodes[node_id]
            coords.append((node["x"], node["y"]))
        if len(coords) >= 2:
            geometries.append(LineString(coords))
    return gpd.GeoDataFrame({"geometry": geometries}, crs="EPSG:4326")


def choose_attack_edges(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ranked = edges_gdf.dropna(subset=["AADT_CUR"]).sort_values("AADT_CUR", ascending=False).copy()
    if ranked.empty:
        return ranked
    ranked = ranked.drop_duplicates(subset=["u", "v"]).head(3).copy()
    ranked["attack_type"] = list(ATTACK_LABELS.keys())[: len(ranked)]
    ranked["attack_label"] = ranked["attack_type"].map(ATTACK_LABELS)
    return ranked


def create_city_figure(
    city_key: str,
    graph: nx.MultiDiGraph,
    edges_gdf: gpd.GeoDataFrame,
    routes_gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 8.8), facecolor="white")
    ax.set_facecolor("white")

    roads = edges_gdf.to_crs(3857)
    roads.plot(ax=ax, color="#d9dee5", linewidth=0.45, alpha=0.95, zorder=1)

    weighted_roads = roads.dropna(subset=["AADT_CUR"])
    local_max_aadt = 1.0
    if not weighted_roads.empty:
        local_max_aadt = max(1.0, float(pd.to_numeric(weighted_roads["AADT_CUR"], errors="coerce").dropna().max()))
    aadt_norm = mpl.colors.Normalize(vmin=0.0, vmax=local_max_aadt)
    if not weighted_roads.empty:
        weighted_roads.plot(
            ax=ax,
            column="AADT_CUR",
            cmap="Blues",
            norm=aadt_norm,
            linewidth=1.7,
            alpha=0.95,
            legend=False,
            zorder=2,
        )

    if not routes_gdf.empty:
        routes_3857 = routes_gdf.to_crs(3857)
        routes_3857.plot(
            ax=ax,
            color="#7b2cbf",
            linewidth=1.15,
            alpha=0.7,
            linestyle="--",
            zorder=3,
        )

    legend_handles = [
        Line2D([0], [0], color="#d9dee5", lw=3, label="Local roads"),
        Line2D([0], [0], color="#4f86c6", lw=3, label="Higher AADT corridors"),
        Line2D([0], [0], color="#7b2cbf", lw=3, linestyle="--", label="Selected routes"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#d6dde6",
        fontsize=9,
    )

    sm = mpl.cm.ScalarMappable(norm=aadt_norm, cmap="Blues")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=35)
    cbar.set_label("AADT", color="#17324d", fontsize=10)
    cbar.ax.tick_params(colors="#17324d", labelsize=8)

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def city_summary(
    city_key: str,
    graph: nx.MultiDiGraph,
    edges_gdf: gpd.GeoDataFrame,
    routes: Sequence[RouteSample],
    attack_edges: gpd.GeoDataFrame,
) -> Dict[str, object]:
    aadt_series = pd.to_numeric(edges_gdf["AADT_CUR"], errors="coerce").dropna()
    lengths_km = []
    for route in routes:
        length_m = 0.0
        for u, v in zip(route.path[:-1], route.path[1:]):
            edge_data = graph.get_edge_data(u, v)
            if not edge_data:
                continue
            first_key = next(iter(edge_data))
            length_m += edge_data[first_key].get("length", 0.0)
        lengths_km.append(length_m / 1000.0)

    return {
        "city": city_key,
        "place": CITY_CONFIGS[city_key]["place"],
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "mean_aadt": round(float(aadt_series.mean()), 2) if not aadt_series.empty else "",
        "max_aadt": int(aadt_series.max()) if not aadt_series.empty else "",
        "routes_sampled": len(routes),
        "mean_route_km": round(sum(lengths_km) / len(lengths_km), 2) if lengths_km else "",
        "attack_edges": len(attack_edges),
    }


def write_summary_table(rows: Sequence[Dict[str, object]], output_csv: Path) -> None:
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_real_city_case_studies(city_keys: Optional[Sequence[str]] = None, seed: int = 42) -> List[Dict[str, object]]:
    ensure_pipeline_directories()
    root = PROJECT_ROOT
    paths = ensure_directories(root)
    selected = list(city_keys or CITY_CONFIGS.keys())
    summaries: List[Dict[str, object]] = []
    cached_city_data = []

    for idx, city_key in enumerate(selected):
        config = CITY_CONFIGS[city_key]
        graph_cache = paths["cache"] / f"{city_key}_drive.graphml"
        graph = download_city_graph(config["place"], config["center"], config["radius_m"], graph_cache)
        _, graph_edges_wgs84 = ox.graph_to_gdfs(graph)
        west, south, east, north = graph_edges_wgs84.total_bounds
        bbox = (west, south, east, north)
        aadt_cache = paths["cache"] / f"{city_key}_txdot_aadt.geojson"
        aadt_gdf = download_txdot_aadt(bbox, aadt_cache)
        if aadt_gdf.crs is None:
            aadt_gdf = aadt_gdf.set_crs(4326)
        else:
            aadt_gdf = aadt_gdf.to_crs(4326)

        edges_gdf = annotate_edges_with_aadt(graph, aadt_gdf)
        weighted_nodes = select_weighted_nodes(graph, edges_gdf)
        routes = sample_routes(graph, weighted_nodes, count=28, seed=seed + idx)
        routes_gdf = route_lines(graph, routes)
        attack_edges = choose_attack_edges(edges_gdf)

        cached_city_data.append((city_key, graph, edges_gdf, routes_gdf, routes, attack_edges))

    for city_key, graph, edges_gdf, routes_gdf, routes, attack_edges in cached_city_data:
        figure_path = paths["figures"] / f"{city_key}_aadt_case_study.png"
        create_city_figure(city_key, graph, edges_gdf, routes_gdf, figure_path)
        summaries.append(city_summary(city_key, graph, edges_gdf, routes, attack_edges))

    write_summary_table(
        summaries,
        paths["generated"] / "real_city_case_summary.csv",
    )
    return summaries


def main() -> None:
    summaries = build_real_city_case_studies()
    for row in summaries:
        print(
            f"{row['city']}: nodes={row['nodes']} edges={row['edges']} "
            f"mean_aadt={row['mean_aadt']} routes={row['routes_sampled']}"
        )


if __name__ == "__main__":
    main()

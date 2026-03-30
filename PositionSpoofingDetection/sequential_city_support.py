from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import math
import random

import networkx as nx
import osmnx as ox
import requests

from real_city_geo import annotate_edges_with_aadt, download_txdot_aadt


def segment_id(u: int, v: int, key: int) -> str:
    return f"{u}|{v}|{key}"


def load_city_graph(city_key: str) -> nx.MultiDiGraph:
    root = Path(__file__).resolve().parents[1]
    return ox.load_graphml(root / "raw_data" / "real_city_cache" / f"{city_key}_drive.graphml")


def load_city_edges(city_key: str, graph: nx.MultiDiGraph):
    import geopandas as gpd

    root = Path(__file__).resolve().parents[1]
    cache = root / "raw_data" / "real_city_cache" / f"{city_key}_txdot_aadt.geojson"
    if cache.exists():
        aadt_gdf = gpd.read_file(cache)
    else:
        _, edges = ox.graph_to_gdfs(graph)
        aadt_gdf = download_txdot_aadt(tuple(edges.total_bounds), cache)
    return annotate_edges_with_aadt(graph, aadt_gdf)


def download_txdot_permanent_count_stations(bounds, cache_path: Path):
    import geopandas as gpd

    if cache_path.exists():
        return gpd.read_file(cache_path)

    minx, miny, maxx, maxy = bounds
    url = (
        "https://services.arcgis.com/KTcxiTD9dsQw4r7Z/ArcGIS/rest/services/"
        "TxDOT_Permanent_Count_Stations/FeatureServer/0/query"
    )
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "geojson",
    }
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    return gpd.read_file(cache_path)


def load_city_count_stations(city_key: str, graph: nx.MultiDiGraph):
    import geopandas as gpd

    root = Path(__file__).resolve().parents[1]
    cache = root / "raw_data" / "real_city_cache" / f"{city_key}_txdot_permanent_count_stations.geojson"
    if cache.exists():
        return gpd.read_file(cache)
    _, edges = ox.graph_to_gdfs(graph)
    return download_txdot_permanent_count_stations(tuple(edges.total_bounds), cache)


def _coerce_aadt(value) -> float:
    try:
        if value is None:
            return 1.0
        numeric = float(value)
        return numeric if math.isfinite(numeric) and numeric > 0 else 1.0
    except Exception:
        return 1.0


def _expand_segments(seed_segments: Sequence[str], adjacency: Dict[str, set[str]], hops: int) -> set[str]:
    frontier = set(seed_segments)
    visited = set(seed_segments)
    for _ in range(max(0, hops)):
        next_frontier: set[str] = set()
        for seg in frontier:
            next_frontier.update(adjacency.get(seg, set()))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def build_taz_context(graph: nx.MultiDiGraph, grid_size: int) -> Dict[int, str]:
    nodes, edges = ox.graph_to_gdfs(graph)
    west, south, east, north = edges.total_bounds
    dx = (east - west) / grid_size
    dy = (north - south) / grid_size

    def zone_of_point(x: float, y: float) -> str:
        ix = min(grid_size - 1, max(0, int((x - west) / dx))) if dx > 0 else 0
        iy = min(grid_size - 1, max(0, int((y - south) / dy))) if dy > 0 else 0
        return f"Z{ix}_{iy}"

    return {int(node_id): zone_of_point(float(row.x), float(row.y)) for node_id, row in nodes.iterrows()}


def build_real_city_context(city_key: str, grid_size: int) -> dict:
    graph = load_city_graph(city_key)
    edges_gdf = load_city_edges(city_key, graph)
    count_stations = load_city_count_stations(city_key, graph)
    node_zone = build_taz_context(graph, grid_size=grid_size)

    zone_production: Dict[str, float] = {}
    zone_attraction: Dict[str, float] = {}
    zone_nodes: Dict[str, List[int]] = {f"Z{ix}_{iy}": [] for ix in range(grid_size) for iy in range(grid_size)}
    segment_to_zone: Dict[str, str] = {}
    segment_to_intersection: Dict[str, str] = {}
    segment_coords: Dict[str, Tuple[float, float]] = {}
    adjacency: Dict[str, set[str]] = {}
    segment_aadt: Dict[str, float] = {}

    for node_id, zone in node_zone.items():
        zone_nodes.setdefault(zone, []).append(node_id)

    for row in edges_gdf.itertuples():
        seg = segment_id(int(row.u), int(row.v), int(row.key))
        zone_u = node_zone[int(row.u)]
        zone_v = node_zone[int(row.v)]
        aadt = _coerce_aadt(getattr(row, "AADT_CUR", None))
        zone_production[zone_u] = zone_production.get(zone_u, 0.0) + aadt
        zone_attraction[zone_v] = zone_attraction.get(zone_v, 0.0) + aadt
        segment_to_zone[seg] = zone_u
        segment_to_intersection[seg] = f"I_{row.v}"
        segment_aadt[seg] = aadt
        geom = getattr(row, "geometry", None)
        if geom is not None:
            mid = geom.interpolate(0.5, normalized=True)
            segment_coords[seg] = (mid.x, mid.y)
        else:
            x1, y1 = float(graph.nodes[int(row.u)]["x"]), float(graph.nodes[int(row.u)]["y"])
            x2, y2 = float(graph.nodes[int(row.v)]["x"]), float(graph.nodes[int(row.v)]["y"])
            segment_coords[seg] = ((x1 + x2) / 2, (y1 + y2) / 2)
        adjacency.setdefault(seg, set())

    by_endpoint: Dict[int, List[str]] = {}
    for seg in segment_coords:
        u, v, _ = seg.split("|")
        by_endpoint.setdefault(int(u), []).append(seg)
        by_endpoint.setdefault(int(v), []).append(seg)
    for connected in by_endpoint.values():
        for seg_a in connected:
            adjacency.setdefault(seg_a, set()).update(seg_b for seg_b in connected if seg_b != seg_a)

    unique_segments = list(segment_coords.keys())

    station_segments: List[str] = []
    if count_stations is not None and not count_stations.empty:
        xs = count_stations.geometry.x.tolist()
        ys = count_stations.geometry.y.tolist()
        nearest = ox.distance.nearest_edges(graph, xs, ys)
        for edge in zip(*nearest) if isinstance(nearest, tuple) else nearest:
            if len(edge) == 3:
                station_segments.append(segment_id(int(edge[0]), int(edge[1]), int(edge[2])))

    if not station_segments:
        sorted_segments = sorted(unique_segments, key=lambda s: segment_coords[s][0])
        anchor_count = 4
        station_segments = [
            sorted_segments[int(i * max(1, len(sorted_segments) - 1) / max(1, anchor_count - 1))]
            for i in range(anchor_count)
        ]

    observed_segments = _expand_segments(station_segments, adjacency, hops=1)
    monitored_intersections = {
        segment_to_intersection[seg]
        for seg in observed_segments
        if seg in segment_to_intersection
    }

    def nearest_station(seg: str) -> str:
        x, y = segment_coords[seg]
        idx = min(
            range(len(station_segments)),
            key=lambda i: (segment_coords[station_segments[i]][0] - x) ** 2
            + (segment_coords[station_segments[i]][1] - y) ** 2,
        )
        return f"R{idx + 1}"

    segment_to_rsu = {seg: nearest_station(seg) for seg in unique_segments}

    aadt_values = sorted(segment_aadt.values())
    median_aadt = aadt_values[len(aadt_values) // 2] if aadt_values else 1.0
    max_aadt = max(aadt_values) if aadt_values else 1.0
    observed_fraction = len(observed_segments) / max(1, len(unique_segments))
    monitored_fraction = len(monitored_intersections) / max(1, len(set(segment_to_intersection.values())))
    message_drop_rate = max(0.02, min(0.30, 0.22 - 0.35 * observed_fraction))
    density_noise_std = max(0.15, min(1.10, 0.90 - 0.90 * observed_fraction))
    vehicle_observation_drop = max(0.0, min(0.15, 0.10 - 0.08 * monitored_fraction))
    controller_delay = 1 if monitored_fraction < 0.12 else 0
    return {
        "graph": graph,
        "edges_gdf": edges_gdf,
        "count_stations": count_stations,
        "zone_production": zone_production,
        "zone_attraction": zone_attraction,
        "zone_nodes": zone_nodes,
        "segment_to_zone": segment_to_zone,
        "segment_to_intersection": segment_to_intersection,
        "segment_to_rsu": segment_to_rsu,
        "segment_coords": segment_coords,
        "adjacency": adjacency,
        "segments": unique_segments,
        "segment_aadt": segment_aadt,
        "observed_segments": observed_segments,
        "monitored_intersections": monitored_intersections,
        "observation_profile": {
            "station_count": len(station_segments),
            "observed_fraction": observed_fraction,
            "monitored_fraction": monitored_fraction,
            "message_drop_rate": message_drop_rate,
            "density_noise_std": density_noise_std,
            "vehicle_observation_drop": vehicle_observation_drop,
            "controller_delay": controller_delay,
            "median_aadt": median_aadt,
            "max_aadt": max_aadt,
        },
    }


def sample_od_pair(context, rng: random.Random) -> Tuple[str, str]:
    origins = list(context["zone_production"].keys())
    origin_weights = [context["zone_production"][z] for z in origins]
    origin_zone = rng.choices(origins, weights=origin_weights, k=1)[0]

    dests = list(context["zone_attraction"].keys())
    dest_weights = [context["zone_attraction"][z] for z in dests]
    for _ in range(30):
        dest_zone = rng.choices(dests, weights=dest_weights, k=1)[0]
        if dest_zone != origin_zone:
            return origin_zone, dest_zone
    return origin_zone, dest_zone


def sample_route_from_od(context, rng: random.Random) -> List[str]:
    graph = context["graph"]
    for _ in range(80):
        origin_zone, dest_zone = sample_od_pair(context, rng)
        origin_candidates = context["zone_nodes"].get(origin_zone, [])
        dest_candidates = context["zone_nodes"].get(dest_zone, [])
        if not origin_candidates or not dest_candidates:
            continue
        origin = rng.choice(origin_candidates)
        destination = rng.choice(dest_candidates)
        if origin == destination:
            continue
        try:
            node_path = nx.shortest_path(graph, origin, destination, weight="travel_time")
        except Exception:
            continue
        route_segments: List[str] = []
        for u, v in zip(node_path[:-1], node_path[1:]):
            bundle = graph.get_edge_data(u, v)
            if not bundle:
                continue
            best_key = min(bundle, key=lambda k: bundle[k].get("travel_time", bundle[k].get("length", 1.0)))
            route_segments.append(segment_id(int(u), int(v), int(best_key)))
        if len(route_segments) >= 4:
            return route_segments
    return []


def build_trip_library(context, seed: int, num_trips: int) -> List[List[str]]:
    rng = random.Random(seed)
    trips: List[List[str]] = []
    progress_step = max(1, num_trips // 5)
    for idx in range(num_trips):
        route = sample_route_from_od(context, rng)
        if route:
            trips.append(route)
        if (idx + 1) % progress_step == 0 or idx + 1 == num_trips:
            print(
                f"[{context['city'].title()}] route library progress: {idx + 1}/{num_trips} candidates, {len(trips)} valid",
                flush=True,
            )
    return trips


def signal_state(tick: int, intersection_id: str) -> str:
    cycle = ("green", "green", "yellow", "red", "red", "red")
    offset = sum(ord(ch) for ch in intersection_id) % len(cycle)
    return cycle[(tick + offset) % len(cycle)]

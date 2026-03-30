from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests


TXDOT_AADT_LAYER = "https://services.arcgis.com/KTcxiTD9dsQw4r7Z/ArcGIS/rest/services/TxDOT_AADT/FeatureServer/0/query"
TXDOT_AADT_FIELD = "AADT_CUR"


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


def annotate_edges_with_aadt(graph: nx.MultiDiGraph, aadt_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _, edges_gdf = ox.graph_to_gdfs(graph)
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

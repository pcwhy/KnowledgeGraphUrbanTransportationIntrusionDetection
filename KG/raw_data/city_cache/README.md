# City Cache Guide

This folder stores the cached city inputs used by the real-city experiments.

## What each file is

For each city, the pipeline expects three cached inputs:

- `<city>_drive.graphml`
  - drivable road network for the city
- `<city>_txdot_aadt.geojson`
  - Annual Average Daily Traffic layer
- `<city>_txdot_permanent_count_stations.geojson`
  - permanent-count-station layer

Current cached cities:

- Austin
- Houston
- Dallas

## Where these files come from

### Drive network

The `*_drive.graphml` files are derived from OpenStreetMap.

Recommended workflow:

1. Use `osmnx` to download the drivable network for the target city.
2. Save the result as GraphML.
3. Store it in this folder with the `<city>_drive.graphml` naming pattern.

The benchmark code expects a drive-network graph rather than an all-mode or
pedestrian graph.

### AADT and count-station layers

The Texas-city pipeline uses TxDOT data for:

- traffic-volume observations
- monitored-station coverage

Recommended workflow for Texas cities:

1. Download the relevant AADT layer from the TxDOT traffic-data or public GIS
   portal.
2. Download the relevant permanent-count-station layer from the same source
   family.
3. Export or convert both layers to GeoJSON.
4. Save them in this folder using:
   - `<city>_txdot_aadt.geojson`
   - `<city>_txdot_permanent_count_stations.geojson`

Recommended workflow outside Texas:

1. Find an equivalent traffic-volume dataset for the target city or region.
2. Find an equivalent monitored-station or count-station dataset.
3. Convert both layers to GeoJSON.
4. Normalize their fields to match what the city benchmark code expects.

## Naming convention

Use one stable city prefix across all three files. Example:

- `austin_drive.graphml`
- `austin_txdot_aadt.geojson`
- `austin_txdot_permanent_count_stations.geojson`

That consistent prefix is the easiest way to register a new city in the
benchmark configuration.

## After adding a new city

Once the three cache files are in place:

1. Register the city in:
   - `experiments/city/real_city_case_study.py`
   - `experiments/city/real_city_benchmark.py`
2. Run the case-study script first to verify that the graph and sensing counts
   look correct.
3. Then run the full city benchmark.

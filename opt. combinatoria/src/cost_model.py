from typing import Dict, List

import numpy as np


def build_total_cost_matrix(
    distance_km: np.ndarray,
    time_h: np.ndarray,
    tolls_mxn: np.ndarray,
    hourly_value_mxn: float,
    vehicle: Dict,
) -> np.ndarray:
    fuel_km_per_l = float(vehicle['fuel_km_per_l'])
    fuel_price = float(vehicle['fuel_price_mxn_per_l'])
    fuel_cost_per_km = fuel_price / fuel_km_per_l

    total = (
        hourly_value_mxn * time_h
        + tolls_mxn
        + distance_km * fuel_cost_per_km
    )
    np.fill_diagonal(total, 0.0)
    return total


def compute_route_cost(route: List[int], cost_matrix: np.ndarray) -> float:
    if len(route) < 2:
        raise ValueError('Route must include at least 2 nodes (start and return).')

    total = 0.0
    for a, b in zip(route[:-1], route[1:]):
        total += float(cost_matrix[a, b])
    return total

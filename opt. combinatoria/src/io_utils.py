import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_config(config_path: Path) -> Dict:
    text = config_path.read_text(encoding='utf-8').strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Config file {config_path} must be valid JSON (JSON is valid YAML 1.2)."
        ) from exc


def load_capitals(capitals_path: Path) -> List[Dict]:
    capitals: List[Dict] = []
    with capitals_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            capitals.append(
                {
                    'state': row['state'],
                    'capital': row['capital'],
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                }
            )
    if not capitals:
        raise ValueError('No capitals were loaded from capitales_32.csv.')
    return capitals


def load_vehicle_catalog(vehicles_path: Path) -> List[Dict]:
    vehicles = json.loads(vehicles_path.read_text(encoding='utf-8'))
    if not isinstance(vehicles, list) or not vehicles:
        raise ValueError('vehiculos.json must contain a non-empty JSON array.')
    return vehicles


def select_vehicle(vehicles: List[Dict], vehicle_id: str) -> Dict:
    for vehicle in vehicles:
        if vehicle.get('vehicle_id') == vehicle_id:
            return vehicle
    available = ', '.join(v.get('vehicle_id', '<missing>') for v in vehicles)
    raise ValueError(f"vehicle_id '{vehicle_id}' not found. Available: {available}")


def load_base_matrices(cost_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: List[Dict] = []
    with cost_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError('costos_matriz.csv is empty.')

    max_idx = max(max(int(r['origin_idx']), int(r['dest_idx'])) for r in rows)
    n = max_idx + 1

    distance = np.zeros((n, n), dtype=float)
    time_h = np.zeros((n, n), dtype=float)
    tolls = np.zeros((n, n), dtype=float)

    for row in rows:
        i = int(row['origin_idx'])
        j = int(row['dest_idx'])
        distance[i, j] = float(row['distance_km'])
        time_h[i, j] = float(row['time_h'])
        tolls[i, j] = float(row['tolls_mxn'])

    return distance, time_h, tolls


def validate_matrix(matrix: np.ndarray, matrix_name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'{matrix_name} must be square.')
    if np.any(matrix < 0):
        raise ValueError(f'{matrix_name} cannot contain negative values.')


def validate_capital_count(capitals: List[Dict], expected: int = 32) -> None:
    if len(capitals) != expected:
        raise ValueError(f'Expected {expected} capitals, got {len(capitals)}.')


def hourly_values_from_config(config: Dict) -> List[float]:
    spec = config.get('hourly_value_mxn', {})
    start = float(spec.get('start', 100.0))
    end = float(spec.get('end', start))
    step = float(spec.get('step', 50.0))
    if step <= 0:
        raise ValueError('hourly_value_mxn.step must be > 0.')
    if end < start:
        raise ValueError('hourly_value_mxn.end must be >= start.')

    values: List[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def find_start_index(capitals: List[Dict], start_capital: str) -> int:
    for idx, item in enumerate(capitals):
        if item['capital'].lower() == start_capital.lower():
            return idx
    names = ', '.join(c['capital'] for c in capitals)
    raise ValueError(f"start_capital '{start_capital}' not found. Available: {names}")

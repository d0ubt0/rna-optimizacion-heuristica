import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import numpy as np

from .aco_solver import solve_aco
from .cost_model import build_total_cost_matrix, compute_route_cost
from .ga_solver import solve_ga
from .io_utils import (
    find_start_index,
    hourly_values_from_config,
    load_base_matrices,
    load_capitals,
    load_config,
    load_vehicle_catalog,
    select_vehicle,
    validate_capital_count,
    validate_matrix,
)
from .types import SolverResult


def _route_to_rows(route: List[int], capitals: List[Dict], cost_matrix: np.ndarray) -> List[Dict]:
    rows: List[Dict] = []
    cumulative = 0.0
    for step, node_idx in enumerate(route):
        capital = capitals[node_idx]
        if step > 0:
            prev_idx = route[step - 1]
            cumulative += float(cost_matrix[prev_idx, node_idx])
        rows.append(
            {
                'step': step,
                'capital': capital['capital'],
                'state': capital['state'],
                'lat': capital['lat'],
                'lon': capital['lon'],
                'cumulative_cost_mxn': round(cumulative, 3),
            }
        )
    return rows


def _write_route_csv(path: Path, route_rows: List[Dict]) -> None:
    if not route_rows:
        raise ValueError('Route rows cannot be empty.')
    fieldnames = list(route_rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(route_rows)


def _write_comparison_csv(path: Path, records: List[Dict]) -> None:
    if not records:
        raise ValueError('Comparison records cannot be empty.')
    fieldnames = list(records[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_experiment(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
) -> Dict:
    config = load_config(config_path)

    capitals = load_capitals(data_dir / 'capitales_32.csv')
    validate_capital_count(capitals, expected=32)

    vehicles = load_vehicle_catalog(data_dir / 'vehiculos.json')
    vehicle = select_vehicle(vehicles, config['vehicle_id'])

    distance_km, time_h, tolls_mxn = load_base_matrices(data_dir / 'costos_matriz.csv')
    validate_matrix(distance_km, 'distance_km')
    validate_matrix(time_h, 'time_h')
    validate_matrix(tolls_mxn, 'tolls_mxn')

    hourly_values = hourly_values_from_config(config)
    start_idx = find_start_index(capitals, config['start_capital'])

    seeds: List[int] = [int(x) for x in config['algorithm_params']['seeds']]
    aco_params: Dict = dict(config['algorithm_params']['aco'])
    ga_params: Dict = dict(config['algorithm_params']['ga'])

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[SolverResult] = []
    matrix_by_hour: Dict[float, np.ndarray] = {}

    for hourly_value in hourly_values:
        total_cost_matrix = build_total_cost_matrix(
            distance_km=distance_km,
            time_h=time_h,
            tolls_mxn=tolls_mxn,
            hourly_value_mxn=hourly_value,
            vehicle=vehicle,
        )
        validate_matrix(total_cost_matrix, f'total_cost_matrix_{hourly_value}')
        matrix_by_hour[hourly_value] = total_cost_matrix

        for seed in seeds:
            aco_result = solve_aco(
                cost_matrix=total_cost_matrix,
                start_idx=start_idx,
                params=aco_params,
                seed=seed,
                hourly_value_mxn=hourly_value,
            )
            ga_result = solve_ga(
                cost_matrix=total_cost_matrix,
                start_idx=start_idx,
                params=ga_params,
                seed=seed,
                hourly_value_mxn=hourly_value,
            )
            all_results.extend([aco_result, ga_result])

    comparison_rows: List[Dict] = []
    for hourly_value in hourly_values:
        for algorithm in ('aco', 'ga'):
            subset = [
                result
                for result in all_results
                if result.algorithm == algorithm
                and abs(result.hourly_value_mxn - hourly_value) < 1e-9
            ]
            costs = [r.total_cost for r in subset]
            best = min(subset, key=lambda x: x.total_cost)

            comparison_rows.append(
                {
                    'hourly_value_mxn': hourly_value,
                    'algorithm': algorithm,
                    'best_cost_mxn': round(min(costs), 4),
                    'mean_cost_mxn': round(mean(costs), 4),
                    'std_cost_mxn': round(pstdev(costs), 4),
                    'best_seed': best.seed,
                    'best_runtime_s': round(best.runtime_s, 4),
                }
            )

    _write_comparison_csv(output_dir / 'comparativa_metricas.csv', comparison_rows)

    best_aco = min((r for r in all_results if r.algorithm == 'aco'), key=lambda x: x.total_cost)
    best_ga = min((r for r in all_results if r.algorithm == 'ga'), key=lambda x: x.total_cost)
    best_global = min(all_results, key=lambda x: x.total_cost)

    aco_matrix = matrix_by_hour[best_aco.hourly_value_mxn]
    ga_matrix = matrix_by_hour[best_ga.hourly_value_mxn]

    _write_route_csv(
        output_dir / 'mejor_ruta_aco.csv',
        _route_to_rows(best_aco.route, capitals, aco_matrix),
    )
    _write_route_csv(
        output_dir / 'mejor_ruta_ga.csv',
        _route_to_rows(best_ga.route, capitals, ga_matrix),
    )

    return {
        'config': config,
        'capitals': capitals,
        'vehicle': vehicle,
        'best_aco': best_aco,
        'best_ga': best_ga,
        'best_global': best_global,
        'best_global_matrix': matrix_by_hour[best_global.hourly_value_mxn],
        'best_global_cost_mxn': compute_route_cost(
            best_global.route, matrix_by_hour[best_global.hourly_value_mxn]
        ),
    }

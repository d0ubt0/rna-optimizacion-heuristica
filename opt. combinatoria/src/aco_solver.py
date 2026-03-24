import time
from typing import Dict, List

import numpy as np

from .cost_model import compute_route_cost
from .types import SolverResult


def solve_aco(
    cost_matrix: np.ndarray,
    start_idx: int,
    params: Dict,
    seed: int,
    hourly_value_mxn: float,
) -> SolverResult:
    n = cost_matrix.shape[0]
    num_ants = int(params.get('num_ants', n))
    iterations = int(params.get('iterations', 100))
    alpha = float(params.get('alpha', 1.0))
    beta = float(params.get('beta', 3.0))
    evaporation = float(params.get('evaporation', 0.3))
    q = float(params.get('q', 100.0))

    if not (0.0 < evaporation < 1.0):
        raise ValueError('ACO evaporation must be in (0,1).')

    rng = np.random.default_rng(seed)
    pheromone = np.ones((n, n), dtype=float)
    visibility = 1.0 / np.maximum(cost_matrix, 1e-12)
    np.fill_diagonal(visibility, 0.0)

    best_cost = float('inf')
    best_route: List[int] = []
    history_best_cost: List[float] = []
    history_best_route: List[List[int]] = []

    t0 = time.perf_counter()

    all_nodes = list(range(n))

    for _ in range(iterations):
        iteration_routes: List[List[int]] = []
        iteration_costs: List[float] = []

        for _ant in range(num_ants):
            route = [start_idx]
            unvisited = [node for node in all_nodes if node != start_idx]
            current = start_idx

            while unvisited:
                desirability = np.array(
                    [
                        (pheromone[current, nxt] ** alpha)
                        * (visibility[current, nxt] ** beta)
                        for nxt in unvisited
                    ],
                    dtype=float,
                )

                if desirability.sum() <= 0:
                    chosen_idx = int(rng.integers(0, len(unvisited)))
                else:
                    probs = desirability / desirability.sum()
                    chosen_idx = int(rng.choice(len(unvisited), p=probs))

                next_city = unvisited.pop(chosen_idx)
                route.append(next_city)
                current = next_city

            route.append(start_idx)
            cost = compute_route_cost(route, cost_matrix)
            iteration_routes.append(route)
            iteration_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_route = route.copy()

        pheromone *= (1.0 - evaporation)
        pheromone = np.maximum(pheromone, 1e-10)

        for route, cost in zip(iteration_routes, iteration_costs):
            delta = q / max(cost, 1e-12)
            for a, b in zip(route[:-1], route[1:]):
                pheromone[a, b] += delta

        history_best_cost.append(float(best_cost))
        history_best_route.append(best_route.copy())

    runtime = time.perf_counter() - t0

    return SolverResult(
        algorithm='aco',
        route=best_route,
        total_cost=float(best_cost),
        history_best_cost=history_best_cost,
        history_best_route=history_best_route,
        iterations=iterations,
        runtime_s=runtime,
        hourly_value_mxn=hourly_value_mxn,
        seed=seed,
    )

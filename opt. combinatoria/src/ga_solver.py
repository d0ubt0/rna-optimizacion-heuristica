import time
from typing import Dict, List

import numpy as np

from .cost_model import compute_route_cost
from .types import SolverResult


def _permutation_to_route(permutation: np.ndarray, start_idx: int) -> List[int]:
    route = [start_idx]
    route.extend(int(x) for x in permutation)
    route.append(start_idx)
    return route


def _tournament_select(
    population: List[np.ndarray],
    fitness: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = rng.choice(len(population), size=tournament_size, replace=False)
    best_local = idx[np.argmin(fitness[idx])]
    return population[int(best_local)].copy()


def _order_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = len(p1)
    a, b = sorted(rng.choice(size, size=2, replace=False))

    child = np.full(size, -1, dtype=int)
    child[a:b] = p1[a:b]

    fill_values = [gene for gene in p2 if gene not in child[a:b]]
    fill_idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = fill_values[fill_idx]
            fill_idx += 1
    return child


def _swap_mutation(individual: np.ndarray, mutation_rate: float, rng: np.random.Generator) -> None:
    if len(individual) < 2:
        return
    if rng.random() < mutation_rate:
        i, j = rng.choice(len(individual), size=2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]


def solve_ga(
    cost_matrix: np.ndarray,
    start_idx: int,
    params: Dict,
    seed: int,
    hourly_value_mxn: float,
) -> SolverResult:
    n = cost_matrix.shape[0]
    cities = [idx for idx in range(n) if idx != start_idx]
    chromosome_len = len(cities)

    population_size = int(params.get('population_size', 120))
    generations = int(params.get('generations', 200))
    crossover_rate = float(params.get('crossover_rate', 0.9))
    mutation_rate = float(params.get('mutation_rate', 0.2))
    elite_size = int(params.get('elite_size', 4))
    tournament_size = int(params.get('tournament_size', 4))

    if elite_size >= population_size:
        raise ValueError('GA elite_size must be smaller than population_size.')

    rng = np.random.default_rng(seed)

    population: List[np.ndarray] = []
    for _ in range(population_size):
        population.append(np.array(rng.permutation(cities), dtype=int))

    def evaluate_population(pop: List[np.ndarray]) -> np.ndarray:
        scores = np.zeros(len(pop), dtype=float)
        for idx, individual in enumerate(pop):
            route = _permutation_to_route(individual, start_idx)
            scores[idx] = compute_route_cost(route, cost_matrix)
        return scores

    t0 = time.perf_counter()

    fitness = evaluate_population(population)
    best_idx = int(np.argmin(fitness))
    best_cost = float(fitness[best_idx])
    best_route = _permutation_to_route(population[best_idx], start_idx)

    history_best_cost: List[float] = []
    history_best_route: List[List[int]] = []

    for _ in range(generations):
        sort_idx = np.argsort(fitness)
        population = [population[int(i)] for i in sort_idx]
        fitness = fitness[sort_idx]

        new_population: List[np.ndarray] = [population[i].copy() for i in range(elite_size)]

        while len(new_population) < population_size:
            parent1 = _tournament_select(population, fitness, tournament_size, rng)
            parent2 = _tournament_select(population, fitness, tournament_size, rng)

            if rng.random() < crossover_rate:
                child = _order_crossover(parent1, parent2, rng)
            else:
                child = parent1.copy()

            _swap_mutation(child, mutation_rate, rng)
            new_population.append(child)

        population = new_population
        fitness = evaluate_population(population)

        gen_best_idx = int(np.argmin(fitness))
        gen_best_cost = float(fitness[gen_best_idx])
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_route = _permutation_to_route(population[gen_best_idx], start_idx)

        history_best_cost.append(best_cost)
        history_best_route.append(best_route.copy())

    runtime = time.perf_counter() - t0

    return SolverResult(
        algorithm='ga',
        route=best_route,
        total_cost=best_cost,
        history_best_cost=history_best_cost,
        history_best_route=history_best_route,
        iterations=generations,
        runtime_s=runtime,
        hourly_value_mxn=hourly_value_mxn,
        seed=seed,
    )

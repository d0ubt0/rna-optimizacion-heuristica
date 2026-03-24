import unittest

import numpy as np

from tsp_mexico.src.aco_solver import solve_aco
from tsp_mexico.src.ga_solver import solve_ga


class SolverRouteValidityTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array(
            [
                [0.0, 10.0, 15.0, 20.0, 12.0],
                [10.0, 0.0, 35.0, 25.0, 18.0],
                [15.0, 35.0, 0.0, 30.0, 16.0],
                [20.0, 25.0, 30.0, 0.0, 14.0],
                [12.0, 18.0, 16.0, 14.0, 0.0],
            ]
        )
        self.start_idx = 0

    def _assert_valid_tsp_route(self, route):
        self.assertEqual(route[0], self.start_idx)
        self.assertEqual(route[-1], self.start_idx)

        visited = route[1:-1]
        self.assertEqual(len(visited), self.matrix.shape[0] - 1)
        self.assertEqual(len(set(visited)), self.matrix.shape[0] - 1)
        self.assertNotIn(self.start_idx, visited)

    def test_aco_route_validity(self):
        result = solve_aco(
            cost_matrix=self.matrix,
            start_idx=self.start_idx,
            params={
                'num_ants': 14,
                'iterations': 30,
                'alpha': 1.0,
                'beta': 3.0,
                'evaporation': 0.3,
                'q': 60.0,
            },
            seed=42,
            hourly_value_mxn=100.0,
        )
        self._assert_valid_tsp_route(result.route)

    def test_ga_route_validity(self):
        result = solve_ga(
            cost_matrix=self.matrix,
            start_idx=self.start_idx,
            params={
                'population_size': 40,
                'generations': 60,
                'crossover_rate': 0.9,
                'mutation_rate': 0.25,
                'elite_size': 3,
                'tournament_size': 3,
            },
            seed=42,
            hourly_value_mxn=100.0,
        )
        self._assert_valid_tsp_route(result.route)


if __name__ == '__main__':
    unittest.main()

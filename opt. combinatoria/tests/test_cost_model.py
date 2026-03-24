import unittest

import numpy as np

from tsp_mexico.src.cost_model import build_total_cost_matrix, compute_route_cost


class CostModelTests(unittest.TestCase):
    def test_total_cost_formula(self):
        distance = np.array([[0.0, 100.0], [100.0, 0.0]])
        time_h = np.array([[0.0, 1.5], [1.5, 0.0]])
        tolls = np.array([[0.0, 80.0], [80.0, 0.0]])
        vehicle = {
            'fuel_km_per_l': 10.0,
            'fuel_price_mxn_per_l': 20.0,
        }

        total = build_total_cost_matrix(
            distance_km=distance,
            time_h=time_h,
            tolls_mxn=tolls,
            hourly_value_mxn=100.0,
            vehicle=vehicle,
        )

        # 100*1.5 + 80 + 100*(20/10) = 430
        self.assertAlmostEqual(total[0, 1], 430.0)
        self.assertAlmostEqual(total[1, 0], 430.0)

    def test_compute_route_cost(self):
        matrix = np.array(
            [
                [0.0, 5.0, 8.0],
                [5.0, 0.0, 2.0],
                [8.0, 2.0, 0.0],
            ]
        )
        route = [0, 1, 2, 0]
        self.assertAlmostEqual(compute_route_cost(route, matrix), 15.0)


if __name__ == '__main__':
    unittest.main()

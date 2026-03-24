from dataclasses import dataclass
from typing import List


@dataclass
class SolverResult:
    algorithm: str
    route: List[int]
    total_cost: float
    history_best_cost: List[float]
    history_best_route: List[List[int]]
    iterations: int
    runtime_s: float
    hourly_value_mxn: float
    seed: int

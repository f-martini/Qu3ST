import numpy as np
import math


def gaussian_integral(x: float, mu: float, sd: float) -> float:
    erf = math.erf((x - mu) / (np.sqrt(2) * sd))
    return erf / 2


def gaussian_mapper(x: float,
                    params: list[tuple[float, float]],
                    delta: float = 0) -> float:
    vals = []
    factors = []
    for mu, sd in params:
        val = (gaussian_integral(x + delta, mu, sd) -
               gaussian_integral(0, mu, sd))
        vals.append(val)
        factor = (gaussian_integral(np.pi, mu, sd) -
                  gaussian_integral(0, mu, sd))
        factors.append(factor)
    val = sum(vals) / sum(factors)
    return val * np.pi - delta


def filter_params(parameters: np.ndarray, sd: float = 0.25) -> np.ndarray:
    params = [(0, sd), (np.pi, sd)]
    delta = np.pi / 2
    new_params = np.vectorize(
        lambda x: gaussian_mapper(x, params=params, delta=delta)
    )(parameters)
    return new_params


def get_bounds_NTS_MD(N: int):
    bounds = []
    for _ in range(N):
        bounds.append((0, 2 * np.pi))

    cons = []
    for factor in range(len(bounds)):
        lower, upper = bounds[factor]
        if lower != - np.inf:
            l = {'type': 'ineq',
                 'fun': lambda x, lb=lower, i=factor: x[i] - lb}
            cons.append(l)
        if upper != np.inf:
            u = {'type': 'ineq',
                 'fun': lambda x, ub=upper, i=factor: ub - x[i]}
            cons.append(u)
    return cons


def get_Z_string(N, idx):
    return "I" * idx + "Z" + "I" * (N - idx - 1)

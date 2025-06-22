import numpy as np


def get_n_qaoa_params(rep, **kwargs):
    return rep * 2


def get_bounds_NTS_QUEEN(Nt, ansatz, **ansatz_options):
    if ansatz == "qaoa":
        N = get_n_qaoa_params(**ansatz_options)
    elif ansatz == "rotations":
        N = Nt
    elif ansatz == "rrrotations":
        N = 3 * Nt
    else:
        raise ValueError(f"{ansatz} is not a valid ansatz.")

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

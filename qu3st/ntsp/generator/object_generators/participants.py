import numpy as np


def generate_participants(t_c):
    n_p = np.random.randint(3, max(t_c // 5, 4)) + 1
    p_d = {
        "own": {p: {"cbs": {}, "sps": {}} for p in range(n_p)},
        "central_banks": [n_p - 1],
        "central_security_depositories": [],
    }
    return p_d

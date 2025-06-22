import numpy as np


def generate_securities(n_sps, scale_lot, scale_price):
    n_sec = np.random.randint(low=1, high=min((int(n_sps) // 5) + 2, 200))
    return [[
        np.random.randint(low=1, high=max(2, scale_lot // 100)),
        float(format(np.random.random() * scale_price + 0.1, '.2f')),
    ] for _ in range(n_sec)]

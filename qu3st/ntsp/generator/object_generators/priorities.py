import numpy as np


def generate_priorities(max_p=16):
    return sorted(
        np.random.random(
            size=np.random.randint(1, max(2, max_p))
        ),
        reverse=True
    )

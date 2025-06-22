import numba as nb
from numba import njit, complex128, int32
import numpy as np


def get_qaoa_initial_params(start="zero", rep=1):
    if start == "zero":
        params = np.zeros(2 * rep)
    elif start == "random":
        params = np.random.rand(2 * rep)
    elif start == "interp":
        params = np.array(
            [-((-1) ** i - 1) / 2 + (-1) ** i * (i * 1 / (rep - 1))
             for i in range(2 * rep)]
        )
    else:
        raise ValueError(f"{start} is not a valid parameter initialization.")
    return params


@njit
def rx(theta):
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return np.array([[cos, -1j * sin],
                     [-1j * sin, cos]])


@njit
def rz(theta):
    exp_u = np.exp(-1.j * theta / 2)
    exp_d = np.exp(1.j * theta / 2)
    return np.array([[exp_u, 0],
                     [0, exp_d]])


@njit
def get_qaoa_counts(angles, weights, N, shots=1000):
    total_outcomes = np.zeros((shots, N))
    probs_0 = np.zeros(N)
    probs_1 = np.zeros(N)
    for i in range(N):
        # define the initial state |+⟩
        final_state = np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]],
                               dtype=nb.Complex)
        # apply all the gates in the i-th wire
        for l in range(len(angles) // (2 * N)):
            theta_1 = angles[2 * l] * weights[i]
            theta_2 = angles[2 * l + 1]

            final_state = rz(theta_1).dot(final_state)
            final_state = rx(theta_2).dot(final_state)
        # calculate the probabilities of each computational basis state
        probs_0[i] = np.abs(final_state[0, 0]) ** 2
        probs_1[i] = np.abs(final_state[1, 0]) ** 2

    for s in range(shots):
        # simulate measurement
        vals = np.random.random(N)
        outcome = (vals < probs_1).astype(int32)
        total_outcomes[s, :] = outcome
    return total_outcomes

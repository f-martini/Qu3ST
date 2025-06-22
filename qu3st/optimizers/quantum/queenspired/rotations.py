import numba as nb
from numba import njit, complex128, int32
import numpy as np


def get_rotation_initial_params(Nt, start="zero", rep=1):
    if start == "one":
        params = np.ones(Nt * rep) * np.pi
    elif start == "zero":
        params = np.zeros(Nt * rep)
    elif start == "plus":
        params = np.ones(Nt * rep) * np.pi / 2
    elif start == "random":
        params = np.random.rand(Nt * rep) * np.pi
    else:
        raise ValueError(
            f"{start} is not a valid parameter initialization.")
    return params


@njit
def rx(theta):
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return np.array([[cos, -1j * sin],
                     [-1j * sin, cos]])


@njit
def ry(theta):
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return np.array([[cos, -sin],
                     [-sin, cos]])


@njit
def rz(theta):
    exp_u = np.exp(-1.j * theta / 2)
    exp_d = np.exp(1.j * theta / 2)
    return np.array([[exp_u, 0],
                     [0, exp_d]])


@njit
def get_rotations_counts(angles: np.ndarray,
                         N: int,
                         shots: int = 1000) -> np.ndarray:
    total_outcomes = np.zeros((shots, N))
    probs_0 = np.zeros(N)
    probs_1 = np.zeros(N)
    for i in range(N):
        theta_1 = angles[0:N][i]

        # define the initial state |0⟩
        initial_state = np.array([[1], [0]], dtype=nb.Complex)
        final_state = rx(theta_1).dot(initial_state)
        # calculate the probabilities of each computational basis state
        probs_0[i] = np.abs(final_state[0, 0]) ** 2
        probs_1[i] = np.abs(final_state[1, 0]) ** 2

    for s in range(shots):
        # simulate measurement
        vals = np.random.random(N)
        outcome = (vals < probs_1).astype(nb.Integer)
        total_outcomes[s, :] = outcome
    return total_outcomes

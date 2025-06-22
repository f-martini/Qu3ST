from numba import njit, complex128, int32
import numpy as np


def get_rrrotation_initial_params(Nt, start="zero", rep=1):
    if start == "one":
        params = np.ones(Nt * 3 * rep) * np.pi
    elif start == "zero":
        params = np.zeros(Nt * 3 * rep)
    elif start == "plus":
        params = np.ones(Nt * 3 * rep) * np.pi / 2
    elif start == "random":
        params = np.random.rand(Nt * 3 * rep)
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
def get_rrrotations_counts(angles, N, shots=1000):
    total_outcomes = np.zeros((shots, N))
    probs_0 = np.zeros(N)
    probs_1 = np.zeros(N)
    for i in range(N):
        theta_1 = angles[0:N][i]
        theta_2 = angles[N:2 * N][i]
        theta_3 = angles[2 * N:3 * N][i]

        probs_0[i] = 1 / 2 * (1 + np.cos(theta_1) * np.cos(theta_3) - np.cos(
            theta_2) * np.sin(theta_1) * np.sin(theta_3))
        probs_1[i] = 1 / 2 * (1 - np.cos(theta_1) * np.cos(theta_3) + np.cos(
            theta_2) * np.sin(theta_1) * np.sin(theta_3))

    for s in range(shots):
        # simulate measurement
        vals = np.random.random(N)
        outcome = (vals < probs_1).astype(int32)
        total_outcomes[s, :] = outcome
    return total_outcomes

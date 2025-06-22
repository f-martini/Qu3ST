from qiskit.circuit import ParameterVector
from .utils import *
from .quantum_circuit import ExtQuantumCircuit as QuantumCircuit


def get_entangling_ladder_ansatz(N, rep=4, barrier=False):
    qc = QuantumCircuit(N, N, name="entangling_ladder")
    qcs = QuantumCircuit(N, N, name="entangling_ladder_scheme")

    beta = ParameterVector("b", N * rep)
    beta_scheme = ParameterVector("bs", N * rep)
    for i in range(rep):
        for j in range(N):
            qc.rx(beta[i * N + j], j)
            qcs.rx(beta_scheme[i * N + j], j)

        for j in range(N - 1):
            qc.cx(j, (j + 1) % N)
            qcs.cx(j, (j + 1) % N)

        add_barrier(qc, barrier)
        add_barrier(qcs, barrier)

    qc.measure(range(N), range(N))
    qcs.measure(range(N), range(N))

    qc.qc_scheme = qcs
    qc.qc_scheme_binder = set_entangling_ladder_scheme

    return qc, beta


def set_entangling_ladder_scheme(qc, H):
    kappa = 1
    ev_t = np.array([1 for i in H])
    param = np.ones(qc.num_parameters) * 2 * np.pi
    return param, kappa, ev_t

from enum import Enum
from typing import Callable

from qiskit.circuit import ParameterVector
from numba import njit, int8
from qiskit.quantum_info import SparsePauliOp, Operator

from .utils import *
from ...objective_function import ObjectiveFunction
from qu3st.optimizers.classical.classical_optimizer import ClassicalOptimizer
from .quantum_circuit import ExtQuantumCircuit as QuantumCircuit

II = ["II", "II"]
XI = ["XI", "XX"]
YI = ["YI", "YX"]
ZI = ["ZI", "ZI"]
IX = ["IX", "IX"]
XX = ["XX", "XI"]
YX = ["YX", "YI"]
ZX = ["ZX", "ZX"]
IY = ["IY", "ZY"]
XY = ["XY", "YZ"]
YY = ["YY", "XZ"]
ZY = ["ZY", "IY"]
IZ = ["IZ", "ZZ"]
XZ = ["XZ", "YY"]
YZ = ["YZ", "XY"]
ZZ = ["ZZ", "IZ"]

bases = np.array(
    [II, XI, YI, ZI, IX, XX, YX, ZX, IY, XY, YY, ZY, IZ, XZ, YZ, ZZ]
)


def apply_pauli(qc, q, op):
    if op == "I":
        pass
    elif op == "X":
        qc.x(q)
    elif op == "Y":
        qc.y(q)
    elif op == "Z":
        qc.z(q)


def twirl(qc, q_c, q_t, idx=None):
    global bases
    idx = np.random.randint(0, 16) if idx is None else idx

    apply_pauli(qc, q_c, bases[idx][0][0])
    apply_pauli(qc, q_t, bases[idx][0][1])

    qc.cx(q_c, q_t)

    apply_pauli(qc, q_c, bases[idx][1][0])
    apply_pauli(qc, q_t, bases[idx][1][1])
    return idx


def get_sm_params(rep, N, random_state=False):
    if random_state:
        params = np.random.rand(rep * N * 3) * np.pi
    else:
        params = np.ones((rep, N, 3)) * np.pi / 2
        params[:, :, 0] = 1.4 / rep
        params[:, :, 2] = 1.4 / rep
        params[0, :, 0] = 0.5 / rep
        params[0, :, 2] = 0.5 / rep
        params = params.flatten()
    return params


def get_sm_ansatz(N, rep=1, barrier=True):
    qc = QuantumCircuit(N, N, name="sm")
    qcs = QuantumCircuit(N, N, name="sm_scheme")

    params = ParameterVector(name="Angles", length=N * 3 * rep)
    # add layers
    for i in range(rep):
        add_barrier(qc, barrier)
        add_barrier(qcs, barrier)
        for q in range(N):
            base_index = i * (N * 3) + q * 3
            if i != rep - 1:
                delta = (np.random.rand() - 0.5) * 0.1
                # add layer to physical quantum circuit
                qc.ry(params[base_index + 0] + delta, q)  # type: ignore
                qc.rz(params[base_index + 1] + delta, q)  # type: ignore
                qc.ry(params[base_index + 2] + delta, q)  # type: ignore

                # add layer to mitigation quantum circuit
                qcs.ry(np.pi, q)
                qcs.rz(np.pi, q)
                qcs.ry(np.pi, q)
            else:
                # add layer to physical quantum circuit
                qc.ry(params[base_index + 0], q)
                qc.rz(params[base_index + 1], q)
                qc.ry(params[base_index + 2], q)

                # add layer to mitigation quantum circuit
                qcs.ry(np.pi, q)
                qcs.rz(np.pi, q)
                qcs.ry(np.pi, q)

        for q in range(1, N):
            add_barrier(qc, barrier)
            add_barrier(qcs, barrier)

            qc.cx(q - 1, q)
            qcs.cx(q - 1, q)

            add_barrier(qc, barrier)
            add_barrier(qcs, barrier)

    qc.measure(range(N), range(N))
    qcs.measure(range(N), range(N))

    qc.qc_scheme = qcs
    qc.qc_scheme_binder = set_sm_ansatz_scheme

    # compute observable
    ops_list = []
    ops_list.append(("Z" * N, 1))
    H = SparsePauliOp.from_list(ops_list)

    return qc, H


def set_sm_ansatz_scheme(qc, H):
    kappa = 1
    ev_t = 1
    return qc, kappa, ev_t

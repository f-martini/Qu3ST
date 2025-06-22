from qiskit.circuit import ParameterVector
from .quantum_circuit import ExtQuantumCircuit as QuantumCircuit


def get_R_ansatz(N, beta=None, generator="X"):
    qc = QuantumCircuit(N, N, name="rotations")
    if beta is None:
        beta = ParameterVector("b", N)
    for i in range(N):
        if generator == "X":
            qc.rx(beta[i], i)
        if generator == "Y":
            qc.ry(beta[i], i)
        if generator == "Z":
            qc.rz(beta[i], i)
    qc.measure(range(N), range(N))
    # print(qc)
    return qc, beta


def get_RRR_ansatz(N, beta=None):
    qc = QuantumCircuit(N, N, name="rrrotations")
    if beta is None:
        beta = ParameterVector("b", 3 * N)
    for i in range(N):
        qc.rx(beta[3 * i], i)
        qc.ry(beta[3 * i + 1], i)
        qc.rz(beta[3 * i + 2], i)
    qc.measure(range(N), range(N))
    # print(qc)
    return qc, beta


def get_base_encoder(N, x):
    qc = QuantumCircuit(N, name="base_encoder")
    for i in range(len(x)):
        if x[i] == 1:
            qc.x(i)
    qc.measure_all()
    return qc

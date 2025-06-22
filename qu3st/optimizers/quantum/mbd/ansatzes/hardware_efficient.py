from .quantum_circuit import ExtQuantumCircuit as QuantumCircuit
from qiskit.circuit import ParameterVector


def get_hardware_efficient(
        num_qubits: int,
        rep: int,
        name: str = "hardware_efficient") -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name=name)
    # add rotation layer
    params = ParameterVector(name="theta", length=num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)

    for l in range(rep):
        start = l % 2
        for i in range(start, num_qubits, 2):
            if i < num_qubits - 1:
                qc.cx(i, i + 1)
    return qc

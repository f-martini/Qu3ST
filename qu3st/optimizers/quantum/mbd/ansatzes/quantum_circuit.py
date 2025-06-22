from typing import Callable, Mapping, Sequence
from qiskit import QuantumCircuit
from qiskit.circuit import Register, Bit
from qiskit.circuit.classical.expr.expr import Var
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameterexpression import ParameterValueType


class ExtQuantumCircuit(QuantumCircuit):

    def __init__(self,
                 *regs: Register | int | Sequence[Bit],
                 qc_scheme: None = None,
                 qc_scheme_binder: Callable | None = None,
                 name: str | None = None,
                 global_phase: ParameterValueType = 0, # type: ignore
                 metadata: dict | None = None, ):
        super().__init__(*regs, name=name,
                         global_phase=global_phase,
                         metadata=metadata)
        self.qc_scheme = qc_scheme if qc_scheme is not None else self
        self.qc_scheme_binder = qc_scheme_binder if qc_scheme_binder is not None else lambda qc, H: qc

    def bind_scheme(self, H):
        return self.qc_scheme_binder(self.qc_scheme, H)

from qiskit.circuit import ParameterVector
from numba import njit, int8
from qiskit.quantum_info import SparsePauliOp

from .utils import *
from .quantum_circuit import ExtQuantumCircuit as QuantumCircuit
from ...objective_function import ObjectiveFunction
from qu3st.optimizers.classical.classical_optimizer import ClassicalOptimizer


def append_x_term(qc, q1, beta):
    qc.rx(beta, q1)


def append_z_term(qc, q1, gamma):
    qc.rz(gamma, q1)


def append_z_pauli_op(qc, op, gamma):
    indexes = [i for i in range(len(op)) if op[i] == "1"]
    if len(indexes) != 0:
        for i in range(1, len(indexes)):
            qc.cx(indexes[i - 1], indexes[i])
        qc.rz(gamma, indexes[-1])
        for i in range(len(indexes) - 1, 0, -1):
            qc.cx(indexes[i - 1], indexes[i])


def get_cost_operator(N, of_weights, const_weights, gamma, name="Cost"):
    qc = QuantumCircuit(N, N, name=name)
    # sum_ofw = np.sum(of_weights)
    for i in range(N):
        if of_weights[i] != 0:
            append_z_term(qc, N - i - 1, of_weights[i] * gamma)  # + sum_ofw)
    for k, val in enumerate(const_weights):
        if val != 0:
            append_z_pauli_op(qc,
                              format(k, '0{}b'.format(N))[::-1],
                              val * gamma)
    return qc


def get_mixer_operator(N, beta, name="Mixer"):
    qc = QuantumCircuit(N, N, name=name)
    for n in range(N):
        append_x_term(qc, n, beta)
    return qc


def get_of_weights(instance, lam):
    of = ObjectiveFunction(instance=instance, lam=lam)
    diag = - of.get_of_weights()
    return diag / 2


@njit
def bin_dot(num1, num2, N):
    # Convert integers to binary arrays
    bin1 = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        bin1[N - 1 - i] = (num1 >> i) & 1
    bin2 = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        bin2[N - 1 - i] = (num2 >> i) & 1
    # Compute the dot product
    dot_product = 0
    for i in range(len(bin1)):
        dot_product += bin1[i] * bin2[i]
    return dot_product


def update_const_weights(const_weights, i, N):
    for k in range(len(const_weights)):
        const_weights[k] += 2 ** -N * (-1) ** bin_dot(i, k, N)
    return const_weights


def get_const_weights(N, instance):
    cl_model = ClassicalOptimizer(
        instance=instance,
        opt_options={
            "verbose": False,
            "collateral": True}
    ).get_model()
    if N > 20:
        raise ValueError(f"Problem size {N} is too large.")
    const_weights = np.zeros(2 ** N, dtype=np.float64)
    for i in range(2 ** N):
        candidate = format(i, '0{}b'.format(N))
        # reverse value to be consistent with states eigenvalues
        candidate = ''.join('1' if bit == '0' else '0' for bit in candidate)
        if not cl_model.is_valid(candidate):
            const_weights = update_const_weights(const_weights, i, N)
    return const_weights


def get_observable(of_weights, const_weights, N):
    pop_list = []
    # generate constraint hamiltonian
    for k, val in enumerate(const_weights):
        pop = format(k, '0{}b'.format(N))
        if val != 0 and k != 0:
            (pop_list.append(
                (''.join(["Z" if po == "1" else "I" for po in pop]), val)
            ))
    # generate objective function hamiltonian
    for k, _ in enumerate(of_weights):
        pop = ["I" for i in range(len(of_weights))]
        pop[k] = 'Z'
        if of_weights[k] != 0:
            pop_list.append((''.join(pop), of_weights[k]))
    H = SparsePauliOp.from_list(pop_list)
    return H


def get_QAOA(N, instance, rep=1, barrier=False, lam=None, obs=False):
    of_weights = get_of_weights(instance, lam)
    beta = ParameterVector("b", rep)
    gamma = ParameterVector("g", rep)
    const_weights = get_const_weights(N, instance)

    qc = QuantumCircuit(N, N, name="QAOA")

    # prepare |+>^N state
    qc.x(range(N))
    qc.h(range(N))
    add_barrier(qc, barrier)

    # add layers
    for i in range(rep):
        add_barrier(qc, barrier)

        if qc is None:
            raise ValueError("Quantum circuit is None. Check the parameters.")
        qc.compose(get_cost_operator(N, of_weights,
                                        const_weights,
                                        beta[i],
                                        name=f"Cost {i}"),
                    qubits=[j for j in range(N)],
                    inplace=True)

        if qc is None:
            raise ValueError("Quantum circuit is None. Check the parameters.")
        qc.compose(get_mixer_operator(N, gamma[i], name=f"Mixer {i}"),
                   qubits=[j for j in range(N)], inplace=True)
        add_barrier(qc, barrier)

    if qc is None:
            raise ValueError("Quantum circuit is None. Check the parameters.")
    qc.measure(range(N), range(N))
    qcs = get_QAOA_scheme(N, rep, const_weights)

    qc.qc_scheme = qcs
    qc.qc_scheme_binder = set_QAOA_scheme
    # print(qc)
    H = None
    if obs:
        H = get_observable(of_weights, const_weights, N)

    return qc, H


def get_QAOA_scheme(N, rep, const_weights):
    qc = QuantumCircuit(N, N, name="QAOA_scheme")

    alpha = ParameterVector("a", rep * N)
    beta = ParameterVector("b", rep * len(const_weights))
    gamma = ParameterVector("g", rep * N)

    # prepare |->^N state
    qc.x(range(N))
    qc.h(range(N))

    # add layers
    for i in range(rep):
        for n in range(N):
            append_z_term(qc,
                          N - n - 1,
                          alpha[rep * i + n])
        for k, val in enumerate(const_weights):
            if val != 0:
                append_z_pauli_op(qc,
                                  format(k, '0{}b'.format(N))[::-1],
                                  beta[rep * i + k])

        for n in range(N):
            append_x_term(qc, n, gamma[rep * i + n])
    qc.measure(range(N), range(N))
    # print(qc)
    return qc


def set_QAOA_scheme(qc, H):
    binded_qc = qc.copy()
    Nq = qc.num_qubits
    # num = np.random.randint(2 ** Nq)
    # bitstring = format(num, '0{}b'.format(Nq))
    bitstring = "1" * Nq

    for p in qc.parameters:
        v = 0
        if "a" in p.name:
            v = np.pi / 2 if p.index < Nq else np.random.rand() + 10 ** -10
        elif "b" in p.name:
            v = 10 ** -10 if p.index < 2 ** Nq else np.random.rand() + 10 ** -10
        elif "g" in p.name:
            if p.index < Nq:
                v = np.pi / 2
                if bitstring[p.index] == "0":
                    v *= -1
            else:
                v = np.pi
        binded_qc.assign_parameters({p: v}, inplace=True)

    if isinstance(H, SparsePauliOp):
        ev_t = H.to_matrix()[0, 0].real
    else:
        ev_t = H({bitstring: 1})
    kappa = 1
    return binded_qc, kappa, ev_t

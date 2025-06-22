from qiskit import QuantumCircuit
from .utils import *
import logging

logger = logging.getLogger(__name__)


def apply_cx(sample, f, s):
    if sample[f] == "1":
        val = "1" if sample[s] == "0" else "0"
        return sample[:s] + val + sample[s + 1:]
    return sample


def get_cx_list_from_circuit(qc: QuantumCircuit) -> list[tuple[int, int]]:
    cx_indexes = []
    for instruction in qc.data:
        if instruction[0].name == "cx":
            pair = (int(qc.find_bit(instruction.qubits[0]).index),
                    int(qc.find_bit(instruction.qubits[1]).index))
            cx_indexes.append(pair)
        elif instruction[0].name == "measure":
            continue
        elif instruction[0].name == "barrier":
            continue
        else:
            if len(cx_indexes) != 0:
                raise TypeError("Circuit structure not compatible with "
                                "self-rem.")
    return cx_indexes


#############################
#    Dep.Chan. whole QC     #
#############################


def get_dep_channel_error_rate(sample: str, probs: dict, nq: int) -> float:
    mit = 0
    if sample in probs.keys():
        mit = probs[sample]
    return 2 ** nq * (1 - mit) / (2 ** nq - 1)


def get_dep_channel_correction(
        key_in: str,
        epsilon: float,
        nq: int,
        probs: dict) -> float:
    val = 0
    for key, p in probs.items():
        if key == key_in:
            val += p * (1 + epsilon * ((1 - 1 / (2 ** nq)) / (1 - epsilon)))
        else:
            val += p * (- epsilon / ((2 ** nq) * (1 - epsilon)))
    return val


def get_dep_channel_mitigated_prob(epsilon: float, nq: int, probs: dict):
    tmp = -epsilon / (2 ** nq)
    tmp *= (epsilon / ((2 ** nq) * (1 - epsilon)))
    tmp *= (2 ** nq - len(probs.keys()))

    tot = 0

    mitigated_probs = {}
    effective_values = {}
    for key, p in probs.items():
        true_p = tmp + get_dep_channel_correction(key, epsilon, nq, probs)
        effective_values[key] = true_p
        mitigated_probs[key] = max(true_p, 0)

        tot += mitigated_probs[key]

    if tot == 0:
        logger.info(effective_values)
        raise ValueError("No positive probs!")
    return {key: v / tot for key, v in mitigated_probs.items()}


#############################
#     Dep.Chan. CX level    #
#############################


def get_cx_error_rate(sample: str, probs: dict, N: int) -> float:
    mit = 0
    if sample in probs.keys():
        mit = probs[sample]
    return 4 * (1 - mit) / (3 * N)


def get_close_samples(sample, cx):
    f = cx[0]
    s = cx[1]
    sample = apply_cx(sample, f, s)
    if f > s:
        tmp = f
        f = s
        s = tmp
    val_nf = "1" if sample[f] == "0" else "0"
    val_ns = "1" if sample[s] == "0" else "0"
    nf = sample[:f] + val_nf + sample[f + 1:]
    ns = sample[:s] + val_ns + sample[s + 1:]
    nfs = sample[:f] + val_nf + sample[f + 1:s] + val_ns + sample[s + 1:]
    return [nf, ns, nfs]


def get_contributor_states(filtered_probs, cx_list):
    Ns = len(cx_list)
    sequences = {k: [] * Ns for k, _ in filtered_probs.items()}
    taus = {k: [""] * Ns for k, _ in filtered_probs.items()}
    # get correct sequence of states
    for tau in filtered_probs.keys():
        tmp = tau
        for i in range(len(cx_list) - 1, -1, -1):
            f = cx_list[i][0]
            s = cx_list[i][1]
            tmp = apply_cx(tmp, f, s)
            taus[tau][i] = tmp

    # logger.info(taus)
    for sample in taus.keys():
        for n, tau in enumerate(taus[sample]):
            valid_close = []
            close_taus = get_close_samples(tau, cx_list[n])
            # logger.info(f"{close_taus} - {cx_list[n]}")

            for close_tau in close_taus:
                tmp = close_tau
                # logger.info(f"{close_tau}")
                for i in range(n + 1, len(cx_list)):
                    f = cx_list[i][0]
                    s = cx_list[i][1]
                    tmp = apply_cx(tmp, f, s)
                    # logger.info(f"{tmp} - {f} {s}")
                if tmp in sequences.keys():
                    valid_close.append(tmp)
            sequences[sample].append(valid_close)
    return sequences


def solve_system(probs: dict, coeff_dict: dict, epsilon: float, Ng: int):
    valid_samples = np.array([k for k in coeff_dict.keys()])
    idxs = {key: n for n, key in enumerate(valid_samples)}
    b = np.array([probs[k] for k in valid_samples])
    a = np.zeros((len(valid_samples), len(valid_samples)))
    main_coeff = 1 - 3 * epsilon * Ng / 4
    sec_coeff = epsilon / 4
    for n, k in enumerate(valid_samples):
        row = idxs[k]
        a[row, row] = main_coeff
        for close_samples in coeff_dict[k]:
            for sample in close_samples:
                a[row, idxs[sample]] += sec_coeff
    solution = np.linalg.solve(a, b)
    # solution -= min(0, min(solution))
    solution[solution < 0] = 0
    solution /= sum(solution)
    return {k: solution[idxs[k]] for k in coeff_dict.keys()}

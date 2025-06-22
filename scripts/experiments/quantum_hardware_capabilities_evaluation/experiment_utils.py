import os
from datetime import datetime
from pathlib import Path
import time
import json
from typing import Any
import pandas as pd
from dotenv import load_dotenv
from qiskit import transpile, qpy
from qiskit.circuit import ParameterVector
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qu3st.optimizers.quantum.mbd.primitives.mitigation import *

import logging

logger = logging.getLogger(__name__)

DATAFRAME_NAME = "hardware_capabilities_data.csv"


def set_base_dict(bits_string: list,
                  prob: float,
                  Nq: int,
                  rotating_indexes: np.ndarray,
                  base_dict: dict) -> None:
    if len(bits_string) == Nq:
        base_dict["".join(bits_string)] = prob
    elif Nq - len(bits_string) - 1 in rotating_indexes:
        set_base_dict(
            bits_string + ["0"], prob * 0.75, Nq, rotating_indexes, base_dict
        )
        set_base_dict(
            bits_string + ["1"], prob * 0.25, Nq, rotating_indexes, base_dict
        )
    else:
        set_base_dict(
            bits_string + ["0"], prob, Nq, rotating_indexes, base_dict
        )


def get_mitigated_probs(qc: QuantumCircuit,
                        mit_sample: str,
                        mit_probs: dict,
                        physical_probs: dict,
                        ) -> tuple:
    Nq = qc.num_qubits

    epsilon_depchannel = get_dep_channel_error_rate(
        mit_sample, mit_probs, Nq
    )

    cx_list = get_cx_list_from_circuit(qc)

    contributors = get_contributor_states(physical_probs, cx_list)
    epsilon = get_cx_error_rate(mit_sample, mit_probs, len(qc))
    logger.info("---")
    logger.info(f"Epsilon cx: {epsilon}")
    logger.info(f"Epsilon depchannel: {epsilon_depchannel}")
    logger.info("---")

    try:
        depchannel_mitigated_probs = get_dep_channel_mitigated_prob(
            epsilon_depchannel, Nq, physical_probs
        )
    except ValueError as e:
        depchannel_mitigated_probs = physical_probs
        logger.info(getattr(e, 'message', str(e)))
        logger.info(f"ValueError raised: ignoring dep. channel mitigation.")

    try:
        mitigated_dist = solve_system(
            physical_probs,
            contributors,
            epsilon,
            len(cx_list)
        )
    except ValueError as e:
        mitigated_dist = physical_probs
        logger.info(getattr(e, 'message', str(e)))
        logger.info(f"ValueError raised: ignoring cx dep. channel mitigation.")

    return depchannel_mitigated_probs, mitigated_dist


def check_row(row: pd.Series, reference_row: dict) -> bool:
    for k in reference_row.keys():
        if not pd.isna(row[k]) and row[k] != reference_row[k]:
            return False
        if pd.isna(row[k]) and reference_row[k] is not None:
            return False
    return True


def load_dataframe(res_path: Path) -> pd.DataFrame:
    res_file = res_path / DATAFRAME_NAME
    if res_file.exists():
        df = pd.read_csv(res_file)
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
    else:
        empty_dict = {
            "backend_name": [],
            "n_states": [],
            "opt_level": [],
            "circuit": [],
            "layers": [],
            "num_qubits": [],
            "conf_id": [],
            "sample_mode": [],
            "shots": [],
            "mitigation_mode": [],
            "info_log": [],
            "res_file": [],
            "fidelity": []
        }
        df = pd.DataFrame(empty_dict)
        df.to_csv(res_file)
    return df


def get_file_name(row: pd.Series) -> str:
    if row["n_states"] is None or pd.isna(row["n_states"]):
        s = ""
    else:
        s = f"{int(row['n_states'])}_"

    return f"{row['backend_name']}_" \
           f"{s}" \
           f"{row['opt_level']}_" \
           f"{row['circuit']}_" \
           f"{row['layers']}_" \
           f"{row['num_qubits']}_" \
           f"{row['conf_id']}_" \
           f"{row['sample_mode']}_" \
           f"{row['shots']}_" \
           f"{row['mitigation_mode']}.json"


def get_qc_name(row: pd.Series) -> str:
    if row["n_states"] is None or pd.isna(row["n_states"]):
        s = ""
    else:
        s = f"{int(row['n_states'])}_"
    return f"{row['backend_name']}_" \
           f"{s}" \
           f"{row['opt_level']}_" \
           f"{row['circuit']}_" \
           f"{row['layers']}_" \
           f"{row['num_qubits']}_" \
           f"{row['conf_id']}.qpy"


def get_row(
        df: pd.DataFrame,
        mitigation_mode: str,
        backend_name: str,
        opt_level: int,
        circuit: str,
        layers: int,
        num_qubits: int,
        conf_id: int,
        sample_mode: str,
        shots: int,
        n_states: int,
        info_log: str) -> tuple[pd.Series, int, bool]:
    partial_row = {
        "mitigation_mode": mitigation_mode,
        "n_states": n_states,
        "backend_name": backend_name,
        "opt_level": opt_level,
        "circuit": circuit,
        "layers": layers,
        "num_qubits": num_qubits,
        "conf_id": conf_id,
        "sample_mode": sample_mode,
        "shots": shots,
    }

    matching_indices = df.loc[
        df.apply(lambda r: check_row(r, partial_row), axis=1)
    ].index

    if len(matching_indices) > 1:
        raise ValueError(f"More then one matching index: {matching_indices}")
    elif len(matching_indices) == 1:
        position = matching_indices[0]
        row = pd.Series(df.loc[position])
        if row["fidelity"] is not None and not pd.isna(row["fidelity"]):
            logger.info(
                f"Row already computed; the old log file can be found at:\n"
                f"\t{row['info_log']}"
            )
            return row, position, True
        else:
            logger.info(
                f"Updating log file; the old log file can be found at:\n"
                f"\t{info_log}"
            )
            row["info_log"] = info_log
            return row, position, False
    else:
        partial_row["info_log"] = info_log
        partial_row["res_file"] = None
        partial_row["fidelity"] = None
        return pd.Series(partial_row), len(df), False


def get_fidelity(A: dict, B: dict) -> float:
    return sum([np.sqrt(A[k] * B[k]) for k in A.keys() if k in B.keys()]) ** 2


def update_row(df: pd.DataFrame,
               row: pd.Series,
               position: int,
               probs: dict,
               ground_truth: dict,
               res_path: Path) -> None:
    row["fidelity"] = get_fidelity(probs, ground_truth)
    df.loc[position] = row
    df.to_csv(res_path / DATAFRAME_NAME)


def run_job(qc: QuantumCircuit,
            params_mit: np.ndarray,
            params_state: np.ndarray,
            shots: int,
            sampler: SamplerV2) -> tuple:
    job = sampler.run(
        [(qc, params_mit), (qc, params_state)],
        shots=shots
    )
    while True:
        if job.status() == 'DONE' or str(job.status()) == 'JobStatus.DONE':
            res_mit = [
                v.get_counts() for v in job.result()[0].data.values()
            ][0]
            res_state = [
                v.get_counts() for v in job.result()[1].data.values()
            ][0]
            break
        time.sleep(10)
    return res_mit, res_state


def get_circuit(
    row: pd.Series,
    qc_path: Path) -> tuple[
        QuantumCircuit, QuantumCircuit, BackendV2, SamplerV2]:
    name = row["circuit"]
    num_qubits = row["num_qubits"]
    layers = row["layers"]
    opt_level = row["opt_level"]
    backend_name = row["backend_name"]

    qc_name = get_qc_name(row)
    backend, sampler = get_primitive(backend_name)

    if (qc_path / qc_name).exists():
        with open(qc_path / qc_name, "rb") as qc_file:
            quantum_circuits = qpy.load(qc_file)
            qc_transpiled = quantum_circuits[0]
            qc = quantum_circuits[1]
        if qc_transpiled.name != qc_name:
            raise ValueError("Wrong circuit name!")
    else:
        qc = QuantumCircuit(num_qubits, name=qc_name)
        # add rotation layer
        params = ParameterVector(name="theta", length=num_qubits)
        for i in range(num_qubits):
            qc.ry(params[i], i)

        for l in range(layers):
            # add cx gate
            if "ladder" == name:
                for i in range(num_qubits - 1):
                    qc.cx(i, i + 1)
            elif "alternating" == name:
                start = l % 2
                for i in range(start, num_qubits, 2):
                    if i < num_qubits - 1:
                        qc.cx(i, i + 1)
            else:
                raise ValueError("Invalid CX insertion mode.")

        qc.measure_all()
        qc_transpiled = transpile(
            qc, backend, optimization_level=opt_level, scheduling_method="asap"
        )

        with open(qc_path / qc_name, "wb") as qc_file:
            qpy.dump([qc_transpiled, qc], qc_file)
    return qc_transpiled, qc, backend, sampler


def apply_cx(sample, f, s):
    if sample[f] == "1":
        val = "1" if sample[s] == "0" else "0"
        return sample[:s] + val + sample[s + 1:]
    return sample


def apply_all_cx(base: str, cx_list: list) -> str:
    base = base[-1: 0: -1] + base[0]
    for cx in cx_list:
        base = apply_cx(base, cx[0], cx[1])
    return base[-1: 0: -1] + base[0]


def get_sparse_params(qc: QuantumCircuit,
                      n_states: int | None = None) -> tuple[dict, np.ndarray]:
    num_qubits = qc.num_qubits
    if n_states is None:
        n_states = num_qubits
    n_rotations = int(np.ceil(np.log2(n_states)))
    rotating_indexes = np.array(random.sample(range(num_qubits), n_rotations))
    base_dict = {}
    set_base_dict([], 1, num_qubits, rotating_indexes, base_dict)
    cx_list = get_cx_list_from_circuit(qc)
    ground_truth = {
        apply_all_cx(k, cx_list): v for k, v in base_dict.items()
    }
    params = np.array([np.pi / 3 if i in rotating_indexes else 0 for i in
                       range(num_qubits)])
    return ground_truth, params


def get_ground_truth(mode: str,
                     qc: QuantumCircuit,
                     n_states: int | None = None) -> tuple[dict, np.ndarray]:
    num_qubits = qc.num_qubits
    if mode == "zero":
        return {"0" * num_qubits: 1}, np.array([0] * num_qubits)
    elif mode == "sparse":
        return get_sparse_params(qc, n_states)
    else:
        raise ValueError("Invalid ground truth mode.")


def print_backend_info(backend: Any) -> None:
    try:
        logger.info(f"Name: {backend.name}\n")
        logger.info(f"Version: {backend.version}\n")
        logger.info(f"No. of qubits: {backend.num_qubits}\n")
        for i in range(backend.num_qubits):
            logger.info(f"Infos qubit {i}: {backend.qubit_properties(i)}")
        logger.info(f"Info instructions: {backend.target}")
    except Exception:
        return


def get_primitive(backend_name: str) -> tuple[BackendV2, SamplerV2]:
    load_dotenv()
    service = QiskitRuntimeService(
        channel=os.getenv('QISKIT_IBM_RUNTIME_CHANNEL'),
        instance=os.getenv('QISKIT_IBM_RUNTIME_INSTANCE'),
        token=os.getenv('QISKIT_API_KEY')
    )
    backend = service.backend(backend_name)
    sampler = SamplerV2(backend)
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    return backend, sampler


def get_counts(row: pd.Series,
               res_path: Path,
               qc_path: Path) -> tuple[
        tuple[dict], tuple[dict], str]:
    file_name = get_file_name(row)
    res_dict = {}
    if (res_path / file_name).exists():
        with open(res_path / file_name, "r") as file:
            res_dict = json.loads(file.read())
    else:
        qc, qc_origin, backend, sampler = get_circuit(row, qc_path)
        ground_truth_mit, params_mit = get_ground_truth("zero", qc_origin)
        ground_truth_sparse, params_sparse = get_ground_truth(
            "sparse", qc_origin, row["n_states"]
        )

        res_dict["samples"] = run_job(
            qc=qc,
            params_mit=params_mit,
            params_state=params_sparse,
            shots=row["shots"],
            sampler=sampler
        )
        print_backend_info(backend)
        res_dict["ground_truth"] = (ground_truth_mit, ground_truth_sparse)
        res_dict["params"] = (
            [float(v) for v in params_mit],
            [float(v) for v in params_sparse]
        )
        with open(res_path / file_name, "w") as file:
            json.dump(res_dict, file)

    return res_dict["samples"], res_dict["ground_truth"], file_name


def normalize_probs(probs: dict) -> dict:
    normalizer = sum(probs.values())
    norm_probs = {k: v / normalizer for k, v in probs.items()}
    return norm_probs


def reduce_probs(raw_probs: dict, max_values: int, shots: int) -> dict:
    additional_samples = {}
    if len(raw_probs) <= max_values:
        return raw_probs
    reduced_sample = raw_probs
    while len(reduced_sample.keys()) > max_values:
        min_prob = min(reduced_sample.values())
        keys = [k for k in reduced_sample.keys()]
        for k in keys:
            if reduced_sample[k] <= min_prob:
                additional_samples[k] = reduced_sample[k]
                reduced_sample.pop(k)
    missing = max_values - len(reduced_sample)
    if missing > 0:
        sampled_keys = random.choices(
            list(additional_samples.keys()),
            weights=list(additional_samples.values()),
            k=missing)
        reduced_sample = {k: v * shots for k, v in reduced_sample.items()}
        for k in sampled_keys:
            if k in reduced_sample.keys():
                reduced_sample[k] += 1
            else:
                reduced_sample[k] = 1
    return normalize_probs(reduced_sample)


def apply_hammer(row: pd.Series,
                 res_path: Path,
                 raw_prob: dict,
                 mit_prob: dict,
                 num_qubits: int,
                 shots: int,
                 callable1: Callable = None,
                 callable2: Callable = None) -> dict:
    file_name = get_file_name(row)
    row["res_file"] = file_name
    if (res_path / file_name).exists():
        with open(res_path / file_name, "r") as file:
            probs = json.loads(file.read())
    else:
        ref_mit_sample = "0" * num_qubits
        ra_error = get_read_out_error(mit_prob, ref_mit_sample)
        max_values = 1000
        if len(raw_prob) > 1000:
            raw_prob = reduce_probs(raw_prob, max_values, shots)
        hammer = Hammer(
            iterative=True,
            iter_weights=False,
            max_hd=num_qubits,
            callable1=callable1,
            callable2=callable2,
        )
        probs = hammer.mitigate(
            ra_error=ra_error,
            prob_error=0 if ref_mit_sample not in mit_prob.keys()
            else 1 - mit_prob[ref_mit_sample],
            probs=raw_prob,
            shots=shots)
        with open(res_path / file_name, "w") as file:
            json.dump(probs, file)
    return probs


def apply_dep_channel_correction(row: pd.Series,
                                 res_path: Path,
                                 num_qubits: int,
                                 mit_sample: str,
                                 mit_probs: dict,
                                 physical_probs: dict,
                                 ) -> dict:
    file_name = get_file_name(row)
    row["res_file"] = file_name
    if (res_path / file_name).exists():
        with open(res_path / file_name, "r") as file:
            depchannel_mitigated_probs = json.loads(file.read())
    else:
        epsilon_depchannel = get_dep_channel_error_rate(
            mit_sample, mit_probs, num_qubits
        )
        logger.info("---")
        logger.info(f"Epsilon depchannel: {epsilon_depchannel}")
        logger.info("---")

        try:
            depchannel_mitigated_probs = get_dep_channel_mitigated_prob(
                epsilon_depchannel, num_qubits, physical_probs
            )
        except ValueError as e:
            depchannel_mitigated_probs = physical_probs
            logger.info(getattr(e, 'message', str(e)))
            logger.info(
                f"ValueError raised: ignoring dep. channel mitigation.")
        with open(res_path / file_name, "w") as file:
            json.dump(depchannel_mitigated_probs, file)
    return depchannel_mitigated_probs


def set_logger(log_path: Path) -> str:
    time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    info_log = f'info_{time_string}.log'
    frmt = '%(asctime)s : %(message)s'
    logging.basicConfig(
        filename=log_path / info_log,
        level=logging.INFO,
        filemode="w",
        format=frmt,
        force=True)
    return info_log

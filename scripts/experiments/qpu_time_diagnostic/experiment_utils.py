import os
from datetime import datetime
from pathlib import Path
import time
import json
from typing import Any
import pandas as pd
from dotenv import load_dotenv
from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qu3st.optimizers.quantum.mbd.primitives.mitigation import *
import logging

logger = logging.getLogger(__name__)


def run_job(qc: QuantumCircuit,
            params_mit: np.ndarray,
            params_state: np.ndarray,
            shots: int,
            sampler: SamplerV2) -> dict:
    job = sampler.run([(qc, params_mit), (qc, params_state)], shots=shots)
    while True:
        if job.status() == 'DONE' or str(job.status()) == 'JobStatus.DONE':
            break
        time.sleep(10)
    metrics = job.metrics()
    tsmp_c = metrics["timestamps"]["created"].replace("Z", "+00:00")
    tsmp_r = metrics["timestamps"]["running"].replace("Z", "+00:00")
    usage = metrics["usage"]["quantum_seconds"]
    time_created = datetime.fromisoformat(tsmp_c)
    time_started = datetime.fromisoformat(tsmp_r)
    time_queue = (time_started - time_created).total_seconds()

    return {
        "id": job.job_id(),
        "created": tsmp_c,
        "device_time": usage,
        "queue_time": time_queue,
        "metrics": job.metrics(),
    }


def get_circuit(row: pd.Series) -> tuple[QuantumCircuit, BackendV2, SamplerV2]:
    name = row["circuit"]
    num_qubits = row["num_qubits"]
    layers = row["layers"]
    opt_level = row["opt_level"]
    backend_name = row["backend_name"]

    backend, sampler = get_primitive(backend_name)

    qc = QuantumCircuit(num_qubits)
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
    return qc_transpiled, backend, sampler


def get_params(qc: QuantumCircuit) -> tuple[np.ndarray, np.ndarray]:
    num_qubits = len(qc.parameters)
    return np.random.rand(num_qubits), np.zeros(num_qubits)


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


def get_file_name(row: pd.Series) -> str:
    return f"{row['backend_name']}_" \
           f"{row['opt_level']}_" \
           f"{row['circuit']}_" \
           f"{row['layers']}_" \
           f"{row['num_qubits']}_" \
           f"{row['conf_id']}_" \
           f"{row['shots']}.json"


def get_times(row: pd.Series, res_path: Path) -> str:
    file_name = get_file_name(row)
    if not (res_path / file_name).exists():
        qc, backend, sampler = get_circuit(row)
        params_mit, params_sparse = get_params(qc)
        res_dict = run_job(
            qc=qc,
            params_mit=params_mit,
            params_state=params_sparse,
            shots=row["shots"],
            sampler=sampler
        )
        print_backend_info(backend)
        with open(res_path / file_name, "w") as file:
            json.dump(res_dict, file)

    return file_name


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


def get_qpu_time(res_path: Path, res_file: str) -> float:
    job_info_file = res_path / res_file

    if not job_info_file.exists():
        raise ValueError("Missing data file.")
    with open(job_info_file, "r") as file:
        job_info = json.load(file)

    if "device_time" in job_info.keys():
        q_time = float(job_info["device_time"])
    else:
        raise ValueError("Missing qpu_time info.")
    return q_time

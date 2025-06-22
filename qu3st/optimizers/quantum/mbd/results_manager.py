import json
from pathlib import Path
from typing import Any

from qiskit import qpy
from qiskit import QuantumCircuit
import logging
import numpy as np
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


class ResultsManager:

    def __init__(self, base_res_path: str | Path):
        if base_res_path is None:
            raise ValueError("A string or path value for the "
                             "result directory must be provided.")
        self.set_results_directory_tree(Path(base_res_path))

    def set_results_directory_tree(self, base_res_path: Path):
        self._base_res_path = Path(base_res_path)
        self._base_res_path.mkdir(parents=True, exist_ok=True)

        self._run_info_path = self.base_res_path / "run_info"
        self._run_info_path.mkdir(parents=True, exist_ok=True)

        self._qc_batch_path = self.base_res_path / "batches"
        self._qc_batch_path.mkdir(parents=True, exist_ok=True)

    @property
    def base_res_path(self):
        return self._base_res_path

    @base_res_path.setter
    def base_res_path(self, value: str | Path):
        if value is None:
            raise ValueError("A string or path value for the "
                             "result directory must be provided.")
        self.set_results_directory_tree(Path(value))

    def save_qcs_info(self, it: int | str,
                      qc: QuantumCircuit,
                      qct: QuantumCircuit | None = None):
        dir_name = self._base_res_path / f"iteration_{it}"
        dir_name.mkdir(parents=True, exist_ok=True)
        with open(dir_name / f"qc_info.qpy", "wb") as file:
            qpy.dump(qc, file)
        if qct is not None:
            with open(dir_name / f"qc_transpiled_info.qpy", "wb") as file:
                qpy.dump(qct, file)

    def save_backend_info(self, it: int | str, backend: Any):
        try:
            dir_name = self._base_res_path / f"iteration_{it}"
            dir_name.mkdir(parents=True, exist_ok=True)
            with open(dir_name / f"{backend.name}_info.txt", "w") as file:
                file.write(f"Name: {backend.name}\n")
                file.write(f"Version: {backend.version}\n")
                file.write(f"No. of qubits: {backend.num_qubits}\n")
                for i in range(backend.num_qubits):
                    file.write(
                        f"Infos qubit {i}: {backend.qubit_properties(i)}")
                file.write(f"Info instructions: {backend.target}")
            logger.info(f"Saved backend info in {backend.name}_info.txt.")
        except Exception:
            return

    def exists_job_data(self, current_it: int, name: str | int = "") -> bool:
        dir_name = self._base_res_path / f"iteration_{current_it}"
        return (dir_name / f'{name}_job_data.json').exists()

    def save_job_data(self, res: dict, current_it: int, name: str | int = ""):
        dir_name = self._base_res_path / f"iteration_{current_it}"
        dir_name.mkdir(parents=True, exist_ok=True)
        with open(dir_name / f'{name}_job_data.json', 'w') as fd:
            res_json = json.dumps(res)
            fd.write(res_json)
        logger.info(f"Saved res to {name}_job_data.json")

    def load_job_data(self, current_it: int, name: str = ""):
        dir_name = self._base_res_path / f"iteration_{current_it}"
        with open(dir_name / f'{name}_job_data.json', 'rb') as handle:
            res = json.load(handle)
        logger.info(f"Loaded res from {name}_job_data.json")
        return res

    def exists_partial_res(self, current_it: int, name: str | int) -> bool:
        dir_name = self._base_res_path / f"iteration_{current_it}"
        return (dir_name / f'{name}_res.json').exists()

    def save_partial_res(self, current_it: int, name: str | int, res: dict):
        dir_name = self._base_res_path / f"iteration_{current_it}"
        dir_name.mkdir(parents=True, exist_ok=True)
        # save qc
        with open(dir_name / f'{name}_res.json', 'w') as fd:
            res_json = json.dumps(res)
            fd.write(res_json)
        logger.info(f"Saved res to {name}_res.json")

    def load_partial_res(self, current_it: int, name: str):
        dir_name = self._base_res_path / f"iteration_{current_it}"
        # load res
        with open(dir_name / f'{name}_res.json', 'rb') as handle:
            res = json.load(handle)
        logger.info(f"Loaded res from {name}_res.json")
        return res

    def exists_run_info(self, name) -> bool:
        if not (self._run_info_path / f'{name}_qc.qpy').exists():
            return False
        if not (self._run_info_path / f'{name}_obs.npy').exists():
            return False
        if not (self._run_info_path / f'{name}_map.npy').exists():
            return False
        return True

    def save_run_info(self,
                      name: str,
                      qc: QuantumCircuit,
                      obs: list,
                      mapping: list):
        # save qc
        with open(self._run_info_path / f'{name}_qc.qpy', 'wb') as fd:
            qpy.dump(qc, fd)
        logger.info(f"Saved qc to {name}_qc.qpy")
        # save pauli op
        obs_list = [(h.paulis.settings["data"][0], h.coeffs[0]) for h in obs]
        np.save(self._run_info_path / f'{name}_obs.npy', obs_list)
        logger.info(f"Saved obs to {name}_obs.npy")
        # save mapping
        np.save(self._run_info_path / f'{name}_map.npy', mapping)
        logger.info(f"Saved mapping to {name}_map.npy")

    def load_run_info(self, name: str):
        # load qc
        with open(self._run_info_path / f'{name}_qc.qpy', 'rb') as handle:
            qc = qpy.load(handle)
        logger.info(f"Loaded qc from {name}_qc.qpy")
        # load pauli op
        obs_list = np.load(self._run_info_path / f'{name}_obs.npy')
        obs = SparsePauliOp.from_list(obs_list)
        logger.info(f"Loaded obs from {name}_obs.qpy")
        # load mapping
        mapping = np.load(self._run_info_path / f'{name}_map.npy')
        logger.info(f"Loaded mapping from {name}_map.npy")
        return qc, obs, mapping

    def exists_qc_batch(self, name: str) -> bool:
        return (self._qc_batch_path / f'{name}_qc.qpy').exists()

    def save_qc_batch(self, name: str, qc: QuantumCircuit):
        with (open(self._qc_batch_path / f'{name}_qc.qpy', 'wb') as fd):
            qpy.dump(qc, fd)

    def load_qc_batch(self, name: str):
        with (open(self._qc_batch_path / f'{name}_qc.qpy', 'rb') as handle):
            qc = qpy.load(handle)
        logger.info(f"Loaded qc from {name}_qc.qpy")
        return qc[0]

    def exists_iter_results(self, current_it: int) -> bool:
        file_name = f"iteration_{current_it}/iter_results.json"
        return (self._base_res_path / file_name).exists()

    def save_iter_results(self,
                          current_solution: dict,
                          ev: float | None,
                          evs: np.ndarray | None,
                          probs: dict | None,
                          parameters: list[np.ndarray],
                          current_it: int):
        to_save = {"best": {"of": current_solution["of"],
                            "solution": current_solution["solution"]},
                   "parameters": [par.tolist() for par in parameters]}
        if ev is not None:
            to_save["ev"] = ev
        if evs is not None:
            to_save["evs"] = evs.tolist()
        if probs is not None:
            to_save["ev"] = ev
        dir_name = self._base_res_path / f"iteration_{current_it}"
        dir_name.mkdir(parents=True, exist_ok=True)
        with open(dir_name / "iter_results.json", "w") as f:
            f.write(json.dumps(to_save))
        logging.info(f"Saved results of iteration {current_it}.")

    def load_iter_results(self, current_it: int) -> dict:
        file_name = f"iteration_{current_it}/iter_results.json"
        with open(self._base_res_path / file_name, 'rb') as handle:
            res = json.load(handle)
        logger.info(f"Loaded results of iteration {current_it}.")
        return res

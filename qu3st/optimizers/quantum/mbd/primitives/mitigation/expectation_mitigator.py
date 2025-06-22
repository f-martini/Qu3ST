import mthree
import time

from multiprocessing import Process, Manager
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import XGate, YGate
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.sampler import SamplerV2
from qiskit_ibm_runtime.transpiler.passes import (PadDynamicalDecoupling,
                                                  ASAPScheduleAnalysis)

from qu3st.optimizers.quantum.mbd.primitives.utils import get_pauli_list, \
    get_expected_values_from_samples

import logging

from qu3st.optimizers.quantum.mbd.results_manager import ResultsManager

logger = logging.getLogger(__name__)
logging.getLogger('qiskit_ibm_runtime').setLevel(logging.ERROR)
logging.getLogger('qiskit.transpiler').setLevel(logging.ERROR)
logging.getLogger('qiskit').setLevel(logging.ERROR)
logging.getLogger('mthree').setLevel(logging.ERROR)

NO_OPT = 0


class ExpectationMitigator:

    def __init__(self,
                 sampler: SamplerV2,
                 mitigation_options: dict,
                 base_pm: PassManager,
                 backend: BackendV2,
                 results_manager: ResultsManager):
        self.sampler = sampler
        self.backend = backend
        self.pm = base_pm
        self.results_manager = results_manager
        self.mitigation_options = {} \
            if mitigation_options is None else mitigation_options
        # dd
        self.pm_dd = self._get_dd_pass_manager(backend)
        # rem
        self.mit = self._get_rem_mit(backend)
        # smit
        self.jobs = self._get_smit_jobs()
        self.name = None
        self.sm_shots = None
        self.sm_ansatz = None
        self.sm_H = None
        self.sm_factor = None
        self.batches = None

        # Calibration
        self.calibration_lifetime = 3600 * 3
        if "calibration_time" in mitigation_options.keys():
            self.calibration_lifetime = mitigation_options["calibration_time"]
        self.rm_calibration_time = None
        self.sm_calibration_time = None

    def _get_rem_mit(self, backend):
        if "rem" in self.mitigation_options.keys():
            if "precomputed" in self.mitigation_options["rem"].keys():
                return self.mitigation_options["rem"]["precomputed"]
            elif "mit" in self.mitigation_options["rem"].keys():
                return mthree.M3Mitigation(backend)
        return None

    def _get_dd_pass_manager(self, backend):
        dd_sequence = None
        if "dd" in self.mitigation_options.keys() \
                and "dd_sequence" in self.mitigation_options["dd"].keys():
            dd_sequence = self.mitigation_options["dd"]["dd_sequence"]

        if dd_sequence is not None:
            gate_sequence = []
            for c in dd_sequence:
                if c == "X":
                    gate = XGate()
                elif c == "Y":
                    gate = YGate()
                else:
                    raise ValueError
                gate_sequence.append(gate)
            return PassManager([
                ASAPScheduleAnalysis(backend.target.durations()),
                PadDynamicalDecoupling(backend.target.durations(),
                                       gate_sequence)
            ])
        else:
            return None

    def _get_smit_jobs(self):
        if "smit" in self.mitigation_options.keys() \
                and "jobs" in self.mitigation_options["smit"].keys():
            return self.mitigation_options["smit"]["jobs"]
        return 1

    def calibrate_rem(self):
        # compute elapsed time
        delta_time = 0
        if self.rm_calibration_time is not None:
            delta_time = time.time() - self.rm_calibration_time

        if "rem" in self.mitigation_options.keys():
            if (self.rm_calibration_time is None or
                    delta_time > self.calibration_lifetime):
                logger.info(f"Calibrating (REM) after {delta_time} seconds.")
                self.rm_calibration_time = time.time()
                if self.mit is None:
                    raise ValueError(
                        "Self-mitigation is not initialized. "
                        "Please set mitigation_options with 'rem' key.")
                self.mit.cals_from_system(range(self.backend.num_qubits))

    def calibrate_smit(self, ansatz, H, twirled_variants, twirled_shots):
        # compute elapsed time
        delta_time = 0
        if self.sm_calibration_time is not None:
            delta_time = time.time() - self.sm_calibration_time

        # self mitigation
        if "smit" in self.mitigation_options.keys():
            if (self.sm_calibration_time is None or
                    delta_time > self.calibration_lifetime or
                    not self._same_experiment(ansatz, H, twirled_variants,
                                              twirled_shots)):
                logger.info(f"Calibrating (SMIT) after {delta_time} seconds.")
                # retrieve quantum circuit scheme, kappa, and expected value
                ansatz_scheme_param, kappa, ev_t = ansatz.bind_scheme(H)
                # set calibration time
                self.sm_calibration_time = time.time()
                # mitigation runs
                results, mappings, obs = self._run_parallel_experiment(
                    self.batches, twirled_shots, None,
                    params=ansatz_scheme_param
                )
                # get count dictionary
                ev_m, mitigation_probs, mitigation_counts, shots_p = (
                    self.process_parallel_jobs_results(
                        results, obs, mappings, twirled_shots
                    )
                )
                # get expected values
                self.sm_factor = (ev_t / ev_m) ** kappa

    def rem_check(self):
        return "rem" in self.mitigation_options.keys()

    def smit_check(self):
        return "smit" in self.mitigation_options.keys()

    def apply_rem(self, raw_probs, mapping):
        if self.mit is None:
            raise ValueError(
                "Self-mitigation is not initialized. "
                "Please set mitigation_options with 'rem' key.")
        return self.mit.apply_correction(raw_probs, mapping)

    def apply_dd(self, qc_final):
        if self.pm_dd is not None:
            qc_dd = self.pm_dd.run(qc_final)
            qc_final = transpile(qc_dd, self.backend,
                                 optimization_level=0)
        return qc_final

    def _run_parallel_experiment(self,
                                 batches,
                                 shots,
                                 res_manager=None,
                                 params=None):
        mps_dict = {}
        obs_dict = {}
        res_dict = {}
        threads = [None] * self.jobs
        jobs_ended = [False] * self.jobs
        for r in range(self.jobs):
            if params is not None:
                batch = [(b[0], params) for b in batches[r]]
            else:
                batch = [b[0] for b in batches[r]]
            obs_dict[r] = [b[1] for b in batches[r]]
            mps_dict[r] = [b[2] for b in batches[r]]
            if res_manager is not None and res_manager.exists_partial_res(r):
                jobs_ended[r] = True
                logger.info(f"Loading pre-coputed results for job {r}.")
                res_dict[r] = res_manager.load_partial_res(r)
            else:
                threads[r] = self.sampler.run(batch, shots=shots)

        while not all(jobs_ended):
            for r, job in enumerate(threads):
                if jobs_ended[r]:
                    continue
                if job.status() == 'DONE' or str(
                        job.status()) == 'JobStatus.DONE':
                    res_dict[r] = [[v.get_counts() for _, v in j.data.items()]
                                   for j in job.result()]
                    if res_manager is not None:
                        res_manager.save_partial_res(r, res_dict[r])
                    else:
                        logger.info("Partial results computed not saved.")
                    logger.info(f"Job-runner {r} DONE!")
                    jobs_ended[r] = True
                elif job.status() == 'ERROR' or job.status() == 'JobStatus.ERROR':
                    logger.error(f' --> {job.status()}\t\t')
                else:
                    logger.info(f"Job-runner {r} waiting: {job.status()}.")
            time.sleep(5)
        tup = ([v for k, v in res_dict.items()],
               [v for k, v in mps_dict.items()],
               [v for k, v in obs_dict.items()])
        return tup

    def _get_expected_values(self, H, mapping, raw_probs):
        obs, coeff = get_pauli_list(H, mapping)
        if "rem" in self.mitigation_options.keys():
            self.calibrate_rem()
            quasis = self.apply_rem(raw_probs, mapping)
            values = quasis.expval(obs)
            values = values * coeff
            raw_probs = quasis.nearest_probability_distribution()
        else:
            # compute expected value list
            values = get_expected_values_from_samples(
                raw_probs, obs, coeff
            )
        return values, raw_probs

    def process_parallel_jobs_results(self, results, obs, mappings, shots):
        evs = []
        raw_counts = {}
        final_probs = {}
        actual_shots = 0
        for r, runs in enumerate(results):
            for u, run in enumerate(runs):
                run_raw_probs = {}
                for k, v in run[0].items():
                    # update raw probs
                    run_raw_probs[k] = v / shots
                    # update raw counts
                    if k in raw_counts.keys():
                        raw_counts[k] += v
                    else:
                        raw_counts[k] = v

                # compute run expected values (optionally mitigate
                # probabilities)
                temp_ev, run_raw_probs = self._get_expected_values(
                    obs[r][u],
                    mappings[r][u],
                    run_raw_probs)
                evs.append(temp_ev)
                actual_shots += shots

                for k, v in run_raw_probs.items():
                    if k in final_probs.keys():
                        final_probs[k] += v
                    else:
                        final_probs[k] = v

        # normalize probabilities
        for k, v in final_probs.items():
            final_probs[k] = v / len(evs)
            raw_counts[k] = raw_counts[k] / actual_shots
        ev = np.average(np.array(evs), axis=0)
        return ev, final_probs, raw_counts, actual_shots

    def _batch_loader(self, i, q, qc, true_size, base_name):
        for s in range(true_size):
            name = base_name + f"_{i}_{s}"
            if self.batch_manager.exists_qc_batch(name):
                qc_final = self.batch_manager.load_qc_batch(name)
            else:
                qc_final = self.pm.run(qc)
                qc_final = transpile(qc_final, self.backend,
                                     optimization_level=NO_OPT,
                                     scheduling_method="asap")
                if self.pm_dd is not None:
                    qc_final = self.pm_dd.run(qc_final)
                    qc_final = transpile(qc_final, self.backend,
                                         optimization_level=NO_OPT)
                self.batch_manager.save_qc_batch(name, qc_final)
            q.put(qc_final)
            print(f"{i} - {s}              ", end="\r")
        logger.info(f"Exiting process {i}.")

    def _get_batches(self, qc, H, variations, size):

        logger.info("Started batch computation...")
        info_name = f"trasnpiled"
        if self.batch_manager.exists_run_info(info_name):
            qc_transpiled, obs, mapping = self.batch_manager.load_run_info(
                info_name
            )
            logger.info("Skipped first transpilation. "
                        "Transpiled circuit loaded from file.")
        else:
            # perform first transpilation
            qc_transpiled = transpile(
                qc, self.backend, optimization_level=3, scheduling_method="asap"
            )
            # generate observable list
            H_list = [SparsePauliOp(ops) for ops in H]
            # mapping logical to physical
            obs = [h.apply_layout(qc_transpiled.layout) for h in H_list]
            mapping = qc_transpiled.layout.final_index_layout()
            self.batch_manager.save_run_info(
                info_name, qc_transpiled, obs, mapping)
            logger.info("Transpiled circuit info saved.")
        threads = []
        batches = []

        with Manager() as manager:
            q = manager.Queue()
            for i in range(0, variations, size):
                true_size = min(size, variations - i)
                t = Process(
                    target=self._batch_loader,
                    args=(i, q, qc_transpiled, true_size, f"batches")
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()
                t.close()
            logger.info("Thread joined")

            while True:
                if q.empty():
                    break
                batch = []
                for i in range(size):
                    if q.empty():
                        break
                    qc = q.get()
                    batch.append((qc, obs, mapping))
                    q.task_done()
                batches.append(batch)
            # print(batches)
            q.join()
        return batches

    def _get_twirled_shots(self, shots, batch):
        twirled_shots = max(1, (shots // self.jobs) // batch)
        if "twirled_shots" in self.mitigation_options["smit"].keys():
            twirled_shots = self.mitigation_options["smit"]["twirled_shots"]
        return twirled_shots

    def _get_batch_size(self):
        # base size
        batch = 300
        # backend base size
        if self.backend.max_circuits is not None:
            batch = self.backend.max_circuits
        # config size
        if "batch" in self.mitigation_options["smit"].keys():
            batch = self.mitigation_options["smit"]["batch"]
        return batch

    def apply_smit(self,
                   ansatz,
                   H,
                   param=None,
                   shots=1000,
                   res_path=None,
                   name=None):
        # set res path
        self.results_manager.res_path = res_path / f"iteration_{name}"
        self.batch_manager.res_path = res_path
        self.name = name

        batch = self._get_batch_size()
        twirled_shots = self._get_twirled_shots(shots, batch)

        # compute total number of different twirled circuits
        twirled_variants = shots // twirled_shots
        if (not self._same_experiment(ansatz, H, twirled_variants,
                                      twirled_shots)
                or self.batches is None):
            self.batches = self._get_batches(
                ansatz, H, twirled_variants, batch)
        else:
            logger.info("Quantum circuit batches already loaded.")
        self.calibrate_smit(
            ansatz, H, twirled_variants, twirled_shots)
        self.sm_ansatz = ansatz
        self.sm_H = H
        self.sm_shots = twirled_variants * twirled_shots
        # physical runs
        results, mappings, obs = self._run_parallel_experiment(
            self.batches, twirled_shots, self.results_manager, params=param
        )
        # get count dictionary and expected values
        ev_p, physical_probs, physical_counts, shots_p = (
            self.process_parallel_jobs_results(results, obs, mappings,
                                               twirled_shots)
        )
        return ev_p * self.sm_factor, physical_counts, physical_probs

    def _same_experiment(self, ansatz, H, twirled_variants, twirled_shots):
        return (ansatz == self.sm_ansatz and
                H == self.sm_H and
                twirled_variants * twirled_shots == self.sm_shots)

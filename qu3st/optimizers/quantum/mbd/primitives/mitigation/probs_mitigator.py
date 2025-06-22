import time
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService

from qu3st.optimizers.quantum.mbd.results_manager import ResultsManager
from .hammer import *
from .utils import *
import logging

from ..utils import add_measurement, update_job_result, get_sampler_job

logger = logging.getLogger(__name__)
logging.getLogger('qiskit_ibm_runtime').setLevel(logging.ERROR)
logging.getLogger('qiskit.transpiler').setLevel(logging.ERROR)
logging.getLogger('qiskit').setLevel(logging.ERROR)
logging.getLogger('mthree').setLevel(logging.ERROR)

NO_OPT = 0


def get_mit_ground_truth(mode: str,
                         qc: QuantumCircuit) -> tuple[dict, np.ndarray, str]:
    num_qubits = qc.num_qubits
    if mode == "zero":
        return ({"0" * num_qubits: 1},
                np.array([0] * num_qubits),
                "0" * num_qubits)
    else:
        raise ValueError("Invalid ground truth mode.")


def run_job(qc: QuantumCircuit,
            params_mit: np.ndarray,
            params_state: np.ndarray,
            shots: int,
            sampler: SamplerV2,
            service: QiskitRuntimeService,
            results_manager: ResultsManager | None,
            it: int | None) -> tuple:

    def retrieve_values(res):
        try:
            values = res.data.values()
        except AttributeError as e:
            logger.info(e)
            values = res["__value__"]["data"].values()
        return values

    job = get_sampler_job(
        service, sampler, [(qc, params_mit), (qc, params_state)], shots,
        results_manager, it)
    results = job.result()
    res_mit = [
        v.get_counts() for v in retrieve_values(results[0])][0]
    res_state = [
        v.get_counts() for v in retrieve_values(results[1])][0]
    logger.info("Physical and mitigation jobs completed!")
    update_job_result(job, results_manager, it, ended=True)
    logger.info("Job info file updated with timestamps.")
    return res_mit, res_state


class ProbsMitigator:

    def __init__(self,
                 sampler: Sampler | SamplerV2,
                 service: QiskitRuntimeService | None,
                 mitigation_options: dict,
                 backend: BackendV2,
                 opt_level: int = 3):
        self.sampler = sampler
        self.backend = backend
        self.service = service
        self.opt_level = opt_level
        self.mitigation_options = {} \
            if mitigation_options is None else mitigation_options
        # rem
        self.rem, self.gt_sample_mode = self._get_rem_mit()

    @staticmethod
    def _get_hammer(iterative: bool = True,
                    iter_weights: bool = False,
                    max_hd: int = 3,
                    mit_sample_mode: str = "zero") -> tuple[Hammer, str]:
        hammer = Hammer(
            iterative=iterative,
            iter_weights=iter_weights,
            max_hd=max_hd
        )
        return hammer, mit_sample_mode

    def _get_rem_mit(self) -> tuple[None | Hammer, None | str]:
        if "rem" in self.mitigation_options.keys():
            if "hammer" in self.mitigation_options["rem"].keys():
                return self._get_hammer(
                    **self.mitigation_options["rem"]["hammer"]
                )
        return None, None

    def rem_check(self) -> bool:
        return ("rem" in self.mitigation_options.keys()
                and len(self.mitigation_options["rem"]) != 0)

    def apply_rem(self,
                  ansatz: QuantumCircuit,
                  results_manager: ResultsManager,
                  params: np.ndarray | None = None,
                  shots: int = 10000,
                  it: int = -1,
                  name: str = "_") -> tuple[dict[str, float], dict[str, float]]:
        # get mitigation params
        if self.gt_sample_mode is None:
            raise ValueError("Ground truth sample mode is not set.")
        ground_truth_mit, params_mit, ref_mit_sample = get_mit_ground_truth(
            self.gt_sample_mode, ansatz
        )
        if (results_manager.exists_partial_res(it, "physical_count") and
                results_manager.exists_partial_res(it, "mitigation_count")):
            physical_count = results_manager.load_partial_res(
                it, "physical_count")
            mitigation_count = results_manager.load_partial_res(
                it, "mitigation_count")
        else:
            # transpile circuit
            add_measurement(ansatz)
            qc_transpiled = transpile(
                ansatz, self.backend,
                optimization_level=self.opt_level,
                scheduling_method="asap"
            )
            # get counts
            mitigation_count, physical_count = run_job(
                qc=qc_transpiled,
                params_mit=params_mit,
                params_state=params,  # type: ignore
                shots=shots,
                sampler=self.sampler,  # type: ignore
                service=self.service,  # type: ignore
                results_manager=results_manager,
                it=it
            )
            results_manager.save_qcs_info(it, ansatz, qc_transpiled)
            results_manager.save_backend_info(it, self.backend)
            results_manager.save_partial_res(it, "physical_count",
                                             physical_count)
            results_manager.save_partial_res(it, "mitigation_count",
                                             mitigation_count)
            logger.info("Saved physical and mitigation results.")

        # process physical count
        if results_manager.exists_partial_res(it, "physical_raw_probs"):
            physical_raw_probs = results_manager.load_partial_res(
                it, "physical_raw_probs"
            )
        else:
            physical_raw_probs = normalize_probs(physical_count)
            if len(physical_raw_probs) > 1000:
                physical_raw_probs = reduce_probs(physical_raw_probs,
                                                  1000,
                                                  shots)
            results_manager.save_partial_res(it,
                                             "physical_raw_probs",
                                             physical_raw_probs)
            logger.info("Saved physical raw probabilities.")

        # compute mitigation data
        if results_manager.exists_partial_res(it, "physical_probs"):
            physical_probs = results_manager.load_partial_res(
                it, "physical_probs"
            )
        else:
            mit_raw_probs = normalize_probs(mitigation_count)
            ra_error = get_read_out_error(mit_raw_probs, ref_mit_sample)
            prob_error = 0
            if ref_mit_sample in mit_raw_probs.keys():
                prob_error = 1 - mit_raw_probs[ref_mit_sample]

            if self.rem is None:
                physical_probs = physical_raw_probs
                logger.info("No REM mitigation applied, using raw physical probs.")
            else:
                physical_probs = self.rem.mitigate(
                    ra_error=ra_error,
                    prob_error=prob_error,
                    probs=physical_raw_probs,
                    shots=shots,
                )
            results_manager.save_partial_res(it,
                                             "physical_probs",
                                             physical_probs)
            logger.info("Saved mitigated physical probabilities.")
        return physical_probs, physical_raw_probs

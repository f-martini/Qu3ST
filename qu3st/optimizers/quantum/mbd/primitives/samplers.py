import os
from qiskit import transpile
from qiskit.primitives import Sampler
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeMelbourneV2
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend
from qiskit_ibm_runtime import SamplerV2
from dotenv import load_dotenv

from .mitigation import ProbsMitigator
from .mitigation.utils import normalize_probs
from .utils import *
from .primitive import Primitive
import logging

from ..results_manager import ResultsManager

logger = logging.getLogger(__name__)


class CustomSampler(Primitive):

    def __init__(self,
                 backend: str | IBMBackend | None,
                 mitigation_options: dict | None = None,
                 backend_options: dict | None = None,
                 **kwargs):
        super().__init__()
        self.mode = backend

        self.mitigation_options = {}
        if mitigation_options is not None:
            self.mitigation_options = mitigation_options

        self.backend_options = {}
        if backend_options is not None:
            self.backend_options = backend_options
        self.service = None
        self.base_sampler, self.backend = self._get_sampler(
            backend, **self.backend_options
        )
        self.mitigator = ProbsMitigator(
            sampler=self.base_sampler,
            service=self.service,
            mitigation_options=mitigation_options,
            backend=self.backend
        )
        if "dd" in self.mitigation_options.keys():
            self._apply_dd(**self.mitigation_options["dd"])
        self.res_path = None
        self.name = None

    def _apply_dd(self, sequence_type: str = "XpXm"):
        self.base_sampler.options.dynamical_decoupling.enable = True
        self.base_sampler.options.dynamical_decoupling.sequence_type \
            = sequence_type

    def _get_real_sampler(self, name: str = "ibm_kyoto"):
        load_dotenv()
        service = QiskitRuntimeService(
            channel=os.getenv('QISKIT_IBM_RUNTIME_CHANNEL'),
            instance=os.getenv('QISKIT_IBM_RUNTIME_INSTANCE'),
            token=os.getenv('QISKIT_API_KEY')
        )
        self.service = service
        try:
            backend = service.backend(name)
        except Exception as e:
            if hasattr(e, 'message'):
                logger.info(e.message)
            else:
                logger.info(e)
            raise ValueError("Invalid real backend")
        return SamplerV2(backend=backend), backend

    @staticmethod
    def _get_fake_sampler(name: str = "fake_sherbrooke"):
        if name == "fake_sherbrooke":
            backend = FakeSherbrooke()
        elif name == "fake_melbourne":
            backend = FakeMelbourneV2()
        else:
            raise ValueError("Invalid fake backend.")
        return SamplerV2(backend=backend), backend

    @staticmethod
    def _get_local_sampler():
        return Sampler(), None

    def _get_sampler(self, backend="local", **backend_options):
        if backend == "local":
            return self._get_local_sampler()
        elif backend == "fake":
            return self._get_fake_sampler(**backend_options)
        elif backend == "real":
            return self._get_real_sampler(**backend_options)
        else:
            raise ValueError(f"{backend} is not a valid backend.")

    def _get_sampler_local_probs(
            self,
            ansatz: QuantumCircuit,
            params: np.ndarray | None = None,
            shots: int | None = None) -> tuple[dict, QuantumCircuit]:
        qsd = self.base_sampler.run(
            ansatz, params, shots=shots
        ).result().quasi_dists[0]
        probs = {format(k, '0{}b'.format(ansatz.num_qubits)): qsd[k] for k in
                 qsd.keys()}
        logger.info(f"Computed local probs -> shots: {shots}.")
        return probs

    def _get_sampler_v2_probs(
            self,
            ansatz: QuantumCircuit,
            params: np.ndarray | None = None,
            shots: int | None = None,
            results_manager: ResultsManager | None = None,
            it: int | None = None) -> dict[str, float]:
        qc = transpile(
            ansatz, self.backend, optimization_level=3, scheduling_method="asap"
        )
        job = get_sampler_job(
            self.service, self.base_sampler, [(qc, params)], shots,
            results_manager, it
        )
        try:
            counts = job.result()[0].data.meas.get_counts()
        except AttributeError as e:
            logger.info(e)
            counts = job.result()[0]["__value__"]["data"].meas.get_counts()
        probs = normalize_probs(counts)
        logger.info(f"Computed probs -> shots: {shots}.")
        if results_manager is not None:
            results_manager.save_qcs_info(it, ansatz, qc)
            results_manager.save_backend_info(it, self.backend)
        return probs

    def _run_experiment(self,
                        ansatz: QuantumCircuit,
                        param: np.ndarray,
                        shots: int,
                        results_manager: ResultsManager,
                        it: int = -1,
                        name: str = "physical_probs") -> dict:
        # prepare circuit
        add_measurement(ansatz)
        # get counts
        if results_manager.exists_partial_res(it, name):
            counts = results_manager.load_partial_res(it, name)
        else:
            if isinstance(self.base_sampler, Sampler):
                counts = self._get_sampler_local_probs(ansatz, param, shots)
            elif isinstance(self.base_sampler, SamplerV2):
                counts = self._get_sampler_v2_probs(
                    ansatz, param, shots, results_manager, it
                )
            else:
                raise ValueError("Sampler type not recognized.")
            results_manager.save_partial_res(it, name, counts)
        return counts

    def run(self,
            ansatz: QuantumCircuit,
            result_manager: ResultsManager,
            param: np.ndarray | None = None,
            shots: int | None = None,
            it: int | None = None,
            name: str | None = None) -> tuple[
        dict[str, float], dict[str, float] | None]:
        """
        Args:
            ansatz: a Qiskit parameterized QuantumCircuit
            param: ansatz parameters
            shots: number of states to sample
            result_manager: ResultsManager instance.
            it: if true sort the QuasiDistribution
            name: the name of the resulting file.
        Returns:
            A dictionary representing the outcome of the sampling process
        """
        # try applying self mitigation
        if self.mitigator.rem_check():
            logger.info("Mitigation active: performing custom mitigation "
                        "pipeline")
            samples, raw_samples = self.mitigator.apply_rem(
                ansatz, result_manager, param, shots,
                it=it, name=name
            )
        else:
            samples = self._run_experiment(
                ansatz, param, shots, results_manager=result_manager, it=it
            )
            raw_samples = None
        return samples, raw_samples

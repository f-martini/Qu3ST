import os
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.fake_provider import FakeMelbourneV2, FakeSherbrooke
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from dotenv import load_dotenv

from .primitive import Primitive
from .mitigation import *
from .utils import *
import logging

from ..results_manager import ResultsManager

logger = logging.getLogger(__name__)
logging.getLogger('qiskit_ibm_runtime').setLevel(logging.ERROR)
logging.getLogger('qiskit.transpiler').setLevel(logging.ERROR)
logging.getLogger('qiskit').setLevel(logging.ERROR)

NO_OPT = 0


class CustomSamplingEstimator(Primitive):

    def __init__(self,
                 backend,
                 mitigation_options=None,
                 callback=None,
                 **kwargs):
        super().__init__()
        # Class general fields
        self.callback = callback
        self.mode = backend
        self.base_sampler, self.backend = self._get_sampler(backend)
        self.pm = PassManager([PauliTwirl()])
        self.samples = {}
        self.max_experiments = None
        self.times = None
        self.name = None
        self.mitigator = ExpectationMitigator(
            sampler=self.base_sampler,
            mitigation_options=mitigation_options,
            base_pm=self.pm,
            backend=self.backend,
            results_manager=ResultsManager(
                base_res_path=""
            )
        )

    @staticmethod
    def _get_real_sampler(name: str = "ibm_kyoto"):
        load_dotenv()
        service = QiskitRuntimeService(
            channel=os.getenv('QISKIT_IBM_RUNTIME_CHANNEL'),
            instance=os.getenv('QISKIT_IBM_RUNTIME_INSTANCE'),
            token=os.getenv('QISKIT_API_KEY')
        )
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

    def _get_sampler(self, backend="fake", **sampler_options):
        if backend == "fake":
            return self._get_fake_sampler(**sampler_options)
        elif backend == "real":
            return self._get_real_sampler(**sampler_options)
        else:
            raise ValueError(f"{backend} is not a valid backend.")

    @staticmethod
    def _validate_circuits_batch(ansatz, mapping, param):
        tups = []
        if isinstance(ansatz, list):
            for n, qc in enumerate(ansatz):
                add_measurement(qc, mapping[n])
                tups = [qc if param is None else (qc, param)]
        else:
            add_measurement(ansatz, mapping)
            tups = [ansatz if param is None else (ansatz, param)]
        return tups

    def _get_expected_values(self, H, mapping, raw_probs):
        obs, coeff = get_pauli_list(H, mapping)
        if self.mitigator.rem_check():
            self.mitigator.calibrate_rem()
            quasis = self.mitigator.apply_rem(raw_probs, mapping)
            values = quasis.expval(obs)
            values = values * coeff
            raw_probs = quasis.nearest_probability_distribution()
        else:
            # compute expected value list
            values = get_expected_values_from_samples(
                raw_probs, obs, coeff
            )
        return values, raw_probs

    def _run_experiment(self, ansatz, H,
                        param=None, shots=None, mapping=None):
        # validate input
        tups = self._validate_circuits_batch(
            ansatz, mapping, param)
        # retrieve results
        job_result = run_job(self.base_sampler, tups, shots)
        if self.callback is not None:
            self.callback(job_result)
        # get count dictionary
        raw_probs, raw_counts, actual_shots = process_results(job_result)
        # get expected values
        values, raw_probs = self._get_expected_values(H, mapping, raw_probs)
        return values, raw_counts, raw_probs

    def _pass_transpile(self, qc, H):
        qc_transpiled = transpile(
            qc, self.backend, optimization_level=0, scheduling_method="asap"
        )
        # generate observable list
        H_list = [SparsePauliOp(ops) for ops in H]
        # mapping logical to physical (NO_OPT=0 allows us to compute it once)
        obs = [h.apply_layout(qc_transpiled.layout) for h in H_list]
        mapping = qc_transpiled.layout.final_index_layout()

        qc_twirled = self.pm.run(qc_transpiled)
        qc_final = transpile(qc_twirled, self.backend,
                             optimization_level=NO_OPT,
                             scheduling_method="asap")
        qc_final = self.mitigator.apply_dd(qc_final)
        return qc_final, obs, mapping

    def run(self,
            ansatz,
            H,
            param=None,
            shots: int | None = None,
            res_path: str | None = None,
            name: str | None = None
            ):
        """
        Args:
            ansatz: a Qiskit parameterized QuantumCircuit
            H: an observable (Hermitian matrix expressed as a SparsePauliOp)
            param: ansatz parameters
            shots: number of states to sample, required parameter for some
            qiskit estimators
            one for each component of H.
            res_path: results path
            name: specific subdirectory for file saving/loading

        Returns:
            A real value resulting from the application of the observable H on
            the circuit
        """

        # try applying self mitigation
        if self.mitigator.smit_check():
            ev, counts, samples = self.mitigator.apply_smit(
                ansatz, H, param, shots, res_path=res_path, name=name
            )
        else:
            qc, obs, mapping = self._pass_transpile(ansatz, H)
            ev, counts, samples = self._run_experiment(
                qc, obs, param, shots, mapping)
        self.samples = samples

        return ev

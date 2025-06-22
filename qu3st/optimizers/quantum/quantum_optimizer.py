from pathlib import Path
import numpy as np
from qu3st.ntsp.instance import Instance
from qu3st.optimizers.result import OptResult
from .mbd.ansatzes.quantum_circuit import ExtQuantumCircuit
from .mbd.primitives.samplers import CustomSampler
from .mbd.primitives.sampling_estimators import CustomSamplingEstimator
from .mbd.results_manager import ResultsManager
from .minimizers.minimizer import Minimizer
from .queenspired.utils import get_bounds_NTS_QUEEN
from ..optimizer import Optimizer
from .mbd.ansatzes import *
from .mbd import *
from .queenspired import *
from .minimizers import *
import logging

logger = logging.getLogger(__name__)


class QuantumOptimizer(Optimizer):
    """
    Implements the NTS-problem solver calling quantum subroutines.
    """

    def __init__(self,
                 instance: Instance,
                 opt_mode: str = "MBD",
                 opt_options: dict | None = None,
                 **kwargs):
        super().__init__(instance,
                         )
        self.opt_mode = opt_mode
        self.opt_options = opt_options if opt_options is not None else {}

    def solve(self, **kwargs) -> OptResult:
        # initialize optimizer and solve NTS
        if self.opt_mode == "MBD":
            logger.info(f"Initializing MBD optimization piepline...")
            return self.solveMBD(**self.opt_options, **kwargs)
        if self.opt_mode == "QUEEN":
            logger.info(f"Initializing QUEEN optimization piepline...")
            return self.solveQUEEN(**self.opt_options, **kwargs)
        else:
            raise ValueError(f"{self.opt_mode} is not a valid optimization "
                             f"method.")

    def get_quantum_circuit(self, ansatz: str, **ansatz_options) \
            -> ExtQuantumCircuit:
        """
        Generate a Qiskit parameterized QuantumCircuit whose ansatz scheme
        can be one of the following: "qaoa", "rotations", "rrrotations",
        "entangling_ladder", "hardware_efficient".
        Args:
            ansatz: a base ansatz scheme
            ansatz_options: ansatz-dependent params
        Returns:
            A parameterized QuantumCircuit
        """
        # generate circuit
        N = len(self.instance.t_cash_amounts())
        if ansatz == "qaoa":
            qc, _ = get_QAOA(N=N, instance=self.instance,
                             **ansatz_options)
        elif ansatz == "rotations":
            qc, _ = get_R_ansatz(N=N, **ansatz_options)
        elif ansatz == "rrrotations":
            qc, _ = get_RRR_ansatz(N=N, **ansatz_options)
        elif ansatz == "entangling_ladder":
            qc, _ = get_entangling_ladder_ansatz(N=N, **ansatz_options)
        elif ansatz == "hardware_efficient":
            qc = get_hardware_efficient(num_qubits=N, **ansatz_options)
        else:
            raise ValueError(f"{ansatz} is not a valid ansatz.")
        logger.info(f"Initialized {ansatz} quantum circuit...")
        logger.info(f"Initialization parameters:\n"
                    f"num_qubits: {N}\n"
                    f"{ansatz_options}")
        return qc

    @staticmethod
    def get_optimizer(mode: str, constraints: list[dict] | None = None, **kwargs) \
            -> Minimizer:
        if mode == "COBYLA":
            opt = COBYLAMinimizer(constraints=constraints, **kwargs)
        elif mode == "ADAM":
            opt = ADAMMinimizer(**kwargs)
        elif mode == "BAYESIAN":
            if constraints is None:
                raise ValueError("Constraints must be provided for Bayesian "
                                 "minimization.")
            bounds = {f'x{i}': (0, np.pi)
                      for i in range(len(constraints) // 2)}
            opt = BayesianMinimizer(bounds, **kwargs)
        else:
            raise ValueError(f"{mode} is not a valid optimizer.")
        logger.info(f"Initialized {mode} optimizer...")
        logger.info(f"Initialization parameters:\n"
                    f"constraints: {constraints}\n"
                    f"{kwargs}")
        return opt

    @staticmethod
    def get_primitive(mode: str, backend: str, **kwargs) \
            -> CustomSampler | CustomSamplingEstimator:
        if mode == "sampler":
            prim = CustomSampler(backend=backend, **kwargs)
        elif mode == "sampling-estimator":
            prim = CustomSamplingEstimator(backend=backend, **kwargs)
        else:
            raise ValueError(f"{mode} is not a valid primitive.")
        logger.info(f"Initialized {mode} primitive...")
        logger.info(
            f"Initialization parameters:\nbackend: {backend}\n{kwargs}")
        return prim

    def solveMBD(self,
                 shots: int | None = None,
                 ansatz: str = "entangling_ladder",
                 ansatz_options: dict | None = None,
                 backend: str | None = None,
                 primitive: str = "sampler",
                 primitive_options: dict | None = None,
                 verbose: bool = False,
                 res_path: str | Path | None = None,
                 gamma: float = 1.0,
                 minimizer: str = "COBYLA",
                 minimizer_options: dict | None = None,
                 of_options: dict | None = None,
                 **kwargs
                 ) -> OptResult:
        if res_path is None:
            res_path = Path("./").resolve()
        logger.info(f"Results for this run will be saved in:\n\t{res_path}")
        results_manager = ResultsManager(res_path)

        if ansatz_options is None:
            ansatz_options = {}
        qc = self.get_quantum_circuit(ansatz, **ansatz_options)

        minimizer_options = minimizer_options if minimizer_options is not None else {}
        opt = self.get_optimizer(
            mode=minimizer,
            constraints=get_bounds_NTS_MD(qc.num_parameters),
            **minimizer_options)

        primitive_options = primitive_options if primitive_options is not None else {}
        prim = self.get_primitive(primitive,
                                  backend=backend if backend is not None else "",
                                  **primitive_options)
        mbd = MBDSolver(
            primitive=prim,
            ansatz=qc,
            results_manager=results_manager,
            shots=shots,
            gamma=gamma,
            optimizer=opt,
            **kwargs
        )
        logger.info(f"MBDSolver initialization completed.")
        return mbd.solve(
            instance=self.instance,
            verbose=verbose,
            of_options=of_options,
            path=res_path,
            **minimizer_options
        )

    def solveQUEEN(self,
                   shots: int = 100,
                   verbose: bool = False,
                   ansatz: str = "rotations",
                   ansatz_options: dict | None = None,
                   res_path: Path | str | None = None,
                   gamma: float = 1.0,
                   minimizer: str = "COBYLA",
                   minimizer_options: dict | None = None,
                   of_options: dict | None = None,
                   **kwargs) -> OptResult:
        if res_path is None:
            res_path = Path("./").resolve()
        logger.info(f"Results for this run will be saved in:\n\t{res_path}")
        results_manager = ResultsManager(res_path)

        opt = self.get_optimizer(
            mode=minimizer,
            constraints=get_bounds_NTS_QUEEN(
                len(self.instance.t_cash_amounts()),
                ansatz,
                **ansatz_options if ansatz_options is not None else {}),
            bounds=None,
            **minimizer_options if minimizer_options is not None else {})

        queen = QueenSpiredSolver(
            instance=self.instance,
            shots=shots,
            optimizer=opt,
            gamma=gamma,
            ansatz=ansatz,
            ansatz_options=ansatz_options,
            results_manager=results_manager,
            **kwargs
        )
        return queen.solve(
            verbose=verbose,
            of_options=of_options,
            path=res_path,
            **minimizer_options if minimizer_options is not None else {}
        )

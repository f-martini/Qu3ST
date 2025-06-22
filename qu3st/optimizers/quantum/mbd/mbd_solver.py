# local imports
from qu3st.ntsp.instance import Instance
from qu3st.utils import VideoGenerator
from qu3st.optimizers.result import OptResult
from .utils import *
from .visualization import *
from .primitives.samplers import *
from .primitives.sampling_estimators import CustomSamplingEstimator
from ..minimizers.minimizer import Minimizer
from ..objective_function import ObjectiveFunction
from ..obs_objective_function import ObsObjectiveFunction

# qiskit imports
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import Optimizer

# general imports
from collections.abc import Callable
from typing import Any, Sequence
from time import time
import logging

logger = logging.getLogger(__name__)


class MBDSolver(SamplingVQE):

    def __init__(
            self,
            primitive: CustomSampler | CustomSamplingEstimator,
            ansatz: QuantumCircuit,
            results_manager: ResultsManager,
            shots: int | None = None,
            gamma: float = 1,
            optimizer: Optimizer | Minimizer | None = None,
            initial_point: Sequence[float] | None = None,
            aggregation: float | Callable[[list[float]], float] | None = None,
            callback: Callable[[int, np.ndarray, float,
                                dict[str, Any]], None] | None = None,
            param_filtering: bool = True,
            **kwargs
    ) -> None:

        super().__init__(
            sampler=Sampler(),
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            aggregation=aggregation,
            callback=callback)
        self.results_manager = results_manager
        self.primitive = primitive
        self.start_time = None
        self.optimizer = optimizer
        self.shots = shots
        self.of = None
        self.verbose = False
        self.best_measurement = None
        self.evaluation_call = 0
        self.gamma = gamma
        self.instance = None
        self.energy_vals = None
        self.Nt = None
        self.current_it = 0
        self.param_filtering = param_filtering
        self.sd = 0.25

    def _init_plot(self, path=None):
        plt.clf()
        self.video_generator = VideoGenerator()
        if path is not None:
            self.video_generator.set_file_name(path, f"MBD_run_{time()}.mp4")
        self.energy_vals = []
        self.fig = plt.figure(figsize=(18, 8))
        gs = self.fig.add_gridspec(1, 1)
        self.plot_1 = self.fig.add_subplot(gs[0, 0])
        self.fig.subplots_adjust(wspace=0.25)

    def _update_plot_loss(self, ev):
        self.energy_vals.append(ev)
        if len(self.energy_vals) % 5 == 0:
            update_plot_loss(energy_vals=self.energy_vals,
                             ax=self.plot_1,
                             title=f"Step 1 (It. {len(self.energy_vals)}): "
                                   f"Optimize All Params - Energy: "
                                   f"{self.energy_vals[-1]} "
                                   f"- Min Energy: {min(self.energy_vals)}",
                             recorder=self.video_generator)

    def _filter_parameters(self, parameters: np.ndarray) -> np.ndarray:
        logger.warning(f"Not Filtered parameters: {parameters}")
        if self.param_filtering:
            parameters = filter_params(parameters, self.sd)
        logger.warning(f"Filtered parameters: {parameters}")
        return parameters

    def _read_partial_results(self,
                              original_parameters,
                              filtered_parameters):

        partial_res = self.results_manager.load_iter_results(
            self.current_it
        )
        self.best_measurement = partial_res["best"]
        origin = np.allclose(np.array(partial_res["parameters"][0]),
                             original_parameters)
        filtered = np.allclose(np.array(partial_res["parameters"][1]),
                               filtered_parameters)
        if not (origin and filtered):
            logger.error(f"Wrong parameters: iter. {self.current_it}.")
            raise ValueError()
        if "ev" in partial_res.keys():
            logger.error(f"ev already computed in partial result! "
                         f"iter. {self.current_it}.")
            raise ValueError("ev already computed in partial result!")

    def _get_evaluate_energy(self,
                             instance,
                             of_options,
                             **kwargs) -> \
            tuple[Callable[[np.ndarray], np.ndarray | float],
                  ObjectiveFunction] | \
            tuple[Callable[[np.ndarray], np.ndarray | float],
                  ObsObjectiveFunction]:

        Nq = self.Nt
        obs = []
        for i in range(Nq):
            obs.append((get_Z_string(Nq, i), 1))
        H = SparsePauliOp.from_list(obs)

        def evaluate_energy_sampling_estimator(parameters: np.ndarray) -> (
                np.ndarray | float):
            filtered_parameters = self._filter_parameters(parameters)
            if self.results_manager.exists_iter_results(self.current_it):
                self._read_partial_results(parameters, filtered_parameters)
            else:
                self.results_manager.save_iter_results(
                    self.best_measurement, None, None, None,
                    [parameters, filtered_parameters], self.current_it
                )
            evs = self.primitive.run(
                self.ansatz, H, parameters,
                shots=self.shots,
                res_path=self.results_manager.base_res_path,
                name=str(self.current_it)
            )
            samples = self.primitive.samples
            ev_dict = {}
            for i, (op, _) in enumerate(H.label_iter()):
                ev_dict[op] = evs[i]
            ev = self.of.evaluate_ev(ev_dict, samples, call=True)
            self.results_manager.save_iter_results(
                self.best_measurement, evs, ev, None,
                [parameters, filtered_parameters], self.current_it
            )
            if self.verbose:
                self._update_plot_loss(ev)
            self.current_it += 1
            self.evaluation_call += 1
            return ev

        def evaluate_energy_sampler(
                parameters: np.ndarray) -> (np.ndarray | float):
            filtered_parameters = self._filter_parameters(parameters)
            if self.results_manager.exists_iter_results(self.current_it):
                self._read_partial_results(parameters, filtered_parameters)
            else:
                self.results_manager.save_iter_results(
                    self.best_measurement, None, None, None,
                    [parameters, filtered_parameters], self.current_it
                )

            final_state, raw_state = self.primitive.run(
                self.ansatz,
                self.results_manager,
                filtered_parameters,
                shots=self.shots,
                it=self.current_it
            )

            logger.info("Evaluating probabilities...")
            probabilities = np.zeros(len(final_state.keys()))
            state = np.zeros((len(final_state.keys()), self.ansatz.num_qubits))
            for n, (k, v) in enumerate(final_state.items()):
                state[n, :] = np.fromiter(k, int)
                probabilities[n] = v
            ev = self.of.evaluate(state, probabilities, gamma=self.gamma)

            if raw_state is not None:
                logger.info("Evaluating raw probabilities...")
                probabilities = np.zeros(len(raw_state.keys()))
                state = np.zeros(
                    (len(raw_state.keys()), self.ansatz.num_qubits))
                for n, (k, v) in enumerate(raw_state.items()):
                    state[n, :] = np.fromiter(k, int)
                    probabilities[n] = v
                _ = self.of.evaluate(state, probabilities, gamma=self.gamma)

            self.results_manager.save_iter_results(
                self.best_measurement, ev, None, final_state,
                [parameters, filtered_parameters], self.current_it
            )
            if self.verbose:
                self._update_plot_loss(ev)
            self.evaluation_call += 1
            self.current_it += 1
            logger.info(f"payoff: {ev}")
            logger.info(f"state (length {len(state)}): {state}\n")
            return ev

        if isinstance(self.primitive, CustomSampler):
            of = ObjectiveFunction(
                instance=instance,
                callback=self.update_best,
                **of_options
            )
            return evaluate_energy_sampler, of
        elif isinstance(self.primitive, CustomSamplingEstimator):
            of = ObsObjectiveFunction(
                instance=instance,
                callback=self.update_best,
                **of_options
            )
            return evaluate_energy_sampling_estimator, of
        else:
            raise TypeError(f"Unexpected type for primitive.")

    def update_best(self, x, ev):
        if self.best_measurement["of"] > ev:
            self.best_measurement["solution"] = x.tolist()
            self.best_measurement["of"] = ev

    def _minimize_wrapper(self, evaluate_energy, max_it=1):

        precomputed_steps = []
        while (self.results_manager.exists_iter_results(self.current_it) and
               len(precomputed_steps) < max_it):
            precomputed_steps.append(
                self.results_manager.load_iter_results(self.current_it)
            )
            self.current_it += 1
        logger.info(f"Loaded {self.current_it} pre-computed iterations.")

        # recompute last iteration
        if (len(precomputed_steps) > 0 and
                "ev" not in precomputed_steps[-1].keys()):
            self.current_it -= 1
            logger.info(f"The last iteration was partially computed: "
                        f"restarting from iteration {self.current_it}).")

        optimizer_result = self.optimizer.minimize(
            fun=evaluate_energy,
            x0=self.initial_point,
            precomputed_steps=precomputed_steps
        )
        return optimizer_result

    def solve(
            self,
            instance: Instance,
            maxiter: int = 1,
            verbose: bool = True,
            sd: float = 0.25,
            of_options: dict | None = None,
            **kwargs
    ) -> OptResult:

        # initialize data
        self.instance = instance
        self.Nt = len(instance.t_cash_amounts())
        self.evaluation_call = 0
        self.current_it = 0
        self.sd = sd

        # get minimizer and objective function
        evaluate_energy, self.of = self._get_evaluate_energy(
            instance, of_options, **kwargs
        )
        self.gamma = self.of.gamma
        self.verbose = verbose
        self.best_measurement = {
            "solution": np.zeros(self.Nt).tolist(),
            "of": self.of.call_evaluate_sample("0" * self.Nt)
        }

        if self.verbose:
            self._init_plot()

        # generate random parameters vector
        self.initial_point = np.random.rand(self.ansatz.num_parameters)
        self.initial_point[:self.ansatz.num_parameters] *= np.pi

        logger.info("Optimization configuration done: starting "
                    "optimization loop...")
        self.start_time = time()
        optimizer_result = self._minimize_wrapper(
            evaluate_energy,
            max_it=maxiter
        )
        optimizer_time = time() - self.start_time
        logger.info("Optimization loop ended.")

        if self.verbose:
            plt.clf()
            self.video_generator.save_video()
        logger.info(
            f"Optimization complete in {optimizer_time} seconds.\n"
        )

        solution = np.array(self.best_measurement["solution"])
        ev, collateral, cb_ind, sp_ind = self.of.call_evaluate_sample(
            solution, coll=True
        )
        return OptResult(
            model=self,
            solver="quantum",
            transactions=np.array([int(bit) for bit in solution]),
            collateral=collateral,
            cb_indicators=cb_ind,
            spl_indicators=sp_ind,
            evaluation=self.of.of(solution),
            evaluation_call=np.inf if self.shots is None
            else self.evaluation_call * self.shots,
            runtime=optimizer_time,
            custom_params={
                "method": "mbd",
                "ansatz": self.ansatz,
                "final_params": optimizer_result.x,
                "instance": instance,
                "energy_vals": self.energy_vals
            })

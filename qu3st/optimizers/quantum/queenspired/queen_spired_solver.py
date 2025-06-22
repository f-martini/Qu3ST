from time import time
from typing import Callable, Any

import numpy as np

from qu3st.ntsp.instance import Instance
from qu3st.utils import VideoGenerator
from qu3st.optimizers.result import OptResult
from qu3st.optimizers.quantum.minimizers.minimizer import Minimizer
from qu3st.optimizers.quantum.objective_function import ObjectiveFunction
from .visualization import *
from .rotations import *
from .rrrotations import *
from .qaoa import *
from ..mbd.utils import filter_params
from qu3st.optimizers.quantum.mbd.results_manager import ResultsManager
import logging

logger = logging.getLogger(__name__)


class QueenSpiredSolver:

    def __init__(self,
                 instance: Instance,
                 optimizer: Minimizer | None = None,
                 shots: int = 1000,
                 verbose: bool = True,
                 ansatz: str = "rotations",
                 ansatz_options: dict | None = None,
                 results_manager: ResultsManager | None = None,
                 param_filtering: bool = True,
                 **kwargs
                 ):
        self.param_filtering = param_filtering
        self.results_manager = results_manager
        self.best_measurement = None
        self.evaluation_call = 0
        self.initial_point = None
        self.of = None
        self.sd = None
        self.optimizer = optimizer
        self.instance = instance
        self.verbose = verbose
        self.shots = shots
        self.ansatz = ansatz
        self.ansatz_options = ansatz_options
        self.kwargs = kwargs

    def _init_plot(self, path=None):
        plt.clf()
        self.video_generator = VideoGenerator()
        if path is not None:
            self.video_generator.set_file_name(path, f"QUEEN_run_{time()}.mp4")
        self.fig = plt.figure(figsize=(18, 8))
        gs = self.fig.add_gridspec(1, 1)
        self.plot_1 = self.fig.add_subplot(gs[0, 0])
        self.fig.subplots_adjust(wspace=0.25)

    def _callback_wrapper(self, it):
        energy_vals = []
        self.current_it = 0

        def _update_plot_loss(params):
            final_state = self.get_counts(
                angles=params, N=len(self.instance.t_cash_amounts()),
                shots=self.shots,
            )

            if self.of is None:
                raise ValueError("Objective function is not set.")

            ev = self.of.evaluate(
                final_state, np.ones(self.shots) / self.shots)
            energy_vals.append(ev)
            if len(energy_vals) % (it // 25) == 0:
                update_plot_loss(energy_vals=energy_vals, ax=self.plot_1,
                                 title=f"Step 1 (It. {len(energy_vals)}): "
                                       f"Optimize All Params - Energy: "
                                       f"{energy_vals[-1]}",
                                 recorder=self.video_generator)

        if self.verbose:
            return _update_plot_loss
        else:
            return None

    def get_counts(self, angles: np.ndarray, N: int, shots: int) -> np.ndarray:
        if self.ansatz == "rotations":
            return get_rotations_counts(
                angles=angles,
                N=N,
                shots=shots)
        elif self.ansatz == "rrrotations":
            return get_rrrotations_counts(
                angles=angles,
                N=N,
                shots=shots)
        elif self.ansatz == "qaoa":
            if self.of is None:
                raise ValueError("Objective function is not set.")
            return get_qaoa_counts(
                angles=angles,
                weights=self.of.of_weights,
                N=N,
                shots=shots)
        else:
            raise ValueError(f"{self.ansatz} is not a valid ansatz.")

    def _filter_parameters(self, parameters: np.ndarray) -> np.ndarray:
        logger.info(f"Not Filtered parameters: {parameters}")
        if self.param_filtering:
            if self.sd is None:
                raise ValueError("Standard deviation (sd) is not set.")
            parameters = filter_params(parameters, self.sd)
        logger.info(f"Filtered parameters: {parameters}")
        return parameters

    def _get_evaluate_energy(self,
                             **kwargs) -> \
            Callable[[np.ndarray], np.ndarray | float] | \
            tuple[Callable[[np.ndarray], np.ndarray | float],
                  dict[str, Any]]:

        def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:
            if self.best_measurement is None:
                raise ValueError("Best measurement is not initialized.")
            if self.of is None:
                raise ValueError("Objective function is not set.")
            if self.results_manager is None:
                raise ValueError("Results manager is not set.")

            filtered_parameters = self._filter_parameters(parameters)
            final_state = self.get_counts(
                angles=filtered_parameters,
                N=len(self.instance.t_cash_amounts()),
                shots=self.shots
            )
            self.results_manager.save_partial_res(
                self.current_it, "physical_probs", final_state.tolist()
            )
            raw_ev = self.of.evaluate(
                final_state, np.ones(self.shots) / self.shots)
            if isinstance(raw_ev, tuple):
                ev = raw_ev[0]
            else:
                ev = raw_ev
            self.results_manager.save_iter_results(
                self.best_measurement, ev, None, final_state.tolist(),
                [parameters, filtered_parameters], self.current_it
            )
            self.current_it += 1
            self.evaluation_call += 1

            return ev

        return evaluate_energy

    def update_best(self, x, ev):
        if self.best_measurement is None:
            raise ValueError("Best measurement is not initialized.")
        if self.best_measurement["of"] > ev:
            self.best_measurement["solution"] = x.tolist()
            self.best_measurement["of"] = ev

    def _minimize_wrapper(self, evaluate_energy, it=1):
        if isinstance(self.optimizer, Minimizer):
            self.optimizer.set_callback(
                callback=self._callback_wrapper(it=it)
            )

        if self.optimizer is None:
            raise ValueError("Optimizer is not set.")
        optimizer_result = self.optimizer.minimize(
            fun=evaluate_energy,
            x0=self.initial_point
        )
        # update initial point
        if optimizer_result is not None:
            self.initial_point = optimizer_result.x
        return optimizer_result

    def get_initial_params(self, Nt):
        if self.ansatz == "qaoa":
            params = get_qaoa_initial_params(
                **self.ansatz_options if self.ansatz_options is not None else {}
            )
        elif self.ansatz == "rrrotations":
            params = get_rrrotation_initial_params(
                Nt, **self.ansatz_options if self.ansatz_options is not None else {}
            )
        elif self.ansatz == "rotations":
            params = get_rotation_initial_params(
                Nt, **self.ansatz_options if self.ansatz_options is not None else {}
            )
        else:
            raise ValueError(
                f"{self.ansatz} is not a valid ansatz.")
        return params

    def solve(
            self,
            maxiter: int = 1,
            verbose=True,
            path=None,
            of_options=None,
            sd: float = 0.25,
            **kwargs
    ) -> OptResult:
        self.sd = sd
        self.verbose = verbose
        # initialize data
        self.of = ObjectiveFunction(
            instance=self.instance,
            callback=self.update_best,
            **of_options if of_options is not None else {}
        )
        self.best_measurement = {
            "solution": np.zeros(len(self.instance.t_cash_amounts())).tolist(),
            "of": self.of.call_evaluate_sample("0" * len(
                self.instance.t_cash_amounts()))
        }
        if self.verbose:
            self._init_plot(path)
        self.evaluation_call = 0

        # generate parameters vector
        self.initial_point = self.get_initial_params(
            len(self.instance.t_cash_amounts()))

        # get minimizer and callback
        evaluate_energy = self._get_evaluate_energy()

        self.start_time = time()

        # optimize both variational and slack variables
        _ = self._minimize_wrapper(evaluate_energy, it=maxiter)
        optimizer_time = time() - self.start_time
        if self.verbose:
            plt.clf()
            self.video_generator.save_video()
            print(
                f"Optimization complete in {optimizer_time} seconds.\n"
            )

        ev, collateral, cb_ind, sp_ind = self.of.call_evaluate_sample(
            self.best_measurement["solution"], coll=True
        )

        solution = self.best_measurement["solution"]
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
                "method": "queen",
                "instance": self.instance,
            })

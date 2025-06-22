import random
from time import time
from pathlib import Path
import numpy as np
from numba import njit

from qu3st.ntsp.instance import Instance
from .. import OptResult
from ..quantum.objective_function import ObjectiveFunction
from ..optimizer import Optimizer
import logging
logger = logging.getLogger(__name__)


@njit
def num2bits(samples: list, bits: int) -> np.ndarray:
    binary_lists = np.zeros((len(samples), bits))
    for n, number in enumerate(samples):
        binary_array = np.zeros(bits, dtype=np.int8)
        for i in range(bits):
            binary_array[bits - 1 - i] = (number >> i) & 1
        binary_lists[n, :] = binary_array
    return binary_lists


class SamplerOptimizer(Optimizer):

    def __init__(self,
                 instance: Instance,
                 opt_options: dict | None = None,
                 **kwargs):
        super().__init__(instance)
        if opt_options is None:
            self.opt_options = {}
        else:
            self.opt_options = opt_options
        self.best_measurement = {
            "solution": np.array([]),
            "of": 0
        }
        self.instance = instance

    def _update_best(self, x: np.ndarray, ev: float):
        if self.best_measurement["of"] > ev:
            self.best_measurement["solution"] = x.tolist()
            self.best_measurement["of"] = ev

    def solve(self,
              res_path: str | None = None,
              **kwargs) -> OptResult:
        n_t = len(self.instance.t_cash_amounts())
        n_candidates = 2 ** n_t
        shots = self.opt_options["shots"]
        iters = self.opt_options["maxiters"]
        n_samples = shots * iters
        of = ObjectiveFunction(
            instance=self.instance,
            callback=self._update_best,
            **self.opt_options["of_options"]
        )
        logger.info(f"Sampling {n_samples} shots")
        start_time = time()
        samples = random.sample(
            range(0, n_candidates),
            min(n_candidates, n_samples)
        )
        samples_bits = num2bits(samples, bits=n_t)
        self.best_measurement = {
            "solution": np.array([0 for _ in range(n_t)]),
            "of": 0
        }
        probabilities = np.zeros(len(samples))
        of.evaluate(samples_bits, probabilities=probabilities)
        optimizer_time = time() - start_time
        solution = np.array(self.best_measurement["solution"])
        ev, collateral, cb_ind, sp_ind = of.call_evaluate_sample(
            solution, coll=True
        )
        opt_res = OptResult(
            model=self,
            solver="sampler",
            transactions=np.array([int(bit) for bit in solution]),
            collateral=collateral,
            cb_indicators=cb_ind,
            spl_indicators=sp_ind,
            evaluation=of.of(solution),
            evaluation_call=-1,
            runtime=optimizer_time,
            custom_params={})
        if res_path is not None:
            opt_res.save(Path(res_path))

        # initialize optimizer and solve NTS
        return opt_res

from typing import Callable
import numpy as np
from qiskit_algorithms.optimizers import ADAM
from qiskit_algorithms.optimizers import OptimizerResult
from qiskit_algorithms.optimizers.optimizer import POINT, Optimizer
from .minimizer import Minimizer


class ADAMMinimizer(ADAM, Minimizer):

    def __init__(
            self,
            maxiter: int = 10000,
            tol: float = 1e-6,
            lr: float = 1e-3,
            beta_1: float = 0.9,
            beta_2: float = 0.99,
            noise_factor: float = 1e-8,
            eps: float = 1e-10,
            amsgrad: bool = False,
            snapshot_dir: str | None = None,
            callback: Callable | None = None,
            **kwargs
    ) -> None:
        super().__init__(maxiter, tol, lr, beta_1, beta_2, noise_factor, eps,
                         amsgrad, snapshot_dir)
        self.callback = callback

    def set_callback(self, callback: Callable):
        self.callback = callback

    def minimize(
            self,
            fun: Callable[[POINT], float],
            x0: POINT,
            jac: Callable[[POINT], POINT] | None = None,
            bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        if jac is None:
            jac = Optimizer.wrap_function(Optimizer.gradient_num_diff,
                                          (fun, self._eps))

        derivative = jac(x0)
        self._t = 0
        self._m = np.zeros(np.shape(derivative))
        self._v = np.zeros(np.shape(derivative))
        if self._amsgrad:
            self._v_eff = np.zeros(np.shape(derivative))

        params = params_new = x0
        while self._t < self._maxiter:
            if self._t > 0:
                derivative = jac(params)
            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (
                    1 - self._beta_2) * derivative * derivative
            lr_eff = self._lr * np.sqrt(1 - self._beta_2 ** self._t) / (
                    1 - self._beta_1 ** self._t)
            if not self._amsgrad:
                params_new = params - lr_eff * self._m.flatten() / (
                        np.sqrt(self._v.flatten()) + self._noise_factor
                )
            else:
                self._v_eff = np.maximum(self._v_eff, self._v)
                params_new = params - lr_eff * self._m.flatten() / (
                        np.sqrt(self._v_eff.flatten()) + self._noise_factor
                )

            if self._snapshot_dir:
                self.save_params(self._snapshot_dir)

            # check termination
            if np.linalg.norm(params - params_new) < self._tol:
                break

            params = params_new
            if self.callback is not None and callable(self.callback):
                self.callback(params)

        result = OptimizerResult()
        result.x = params_new
        result.fun = fun(params_new)
        result.nfev = self._t
        return result

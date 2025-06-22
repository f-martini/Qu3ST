from typing import Callable
from qiskit_algorithms.optimizers import COBYLA, OptimizerResult
from qiskit_algorithms.optimizers.optimizer import POINT, Optimizer
from .minimizer import Minimizer
import numpy as np
from scipy.optimize import minimize


class COBYLAMinimizer(COBYLA, Minimizer):

    def __init__(self,
                 maxiter: int = 1000,
                 disp: bool = False,
                 rhobeg: float = 1.0,
                 tol: float | None = None,
                 options: dict | None = None,
                 callback: Callable | None = None,
                 **kwargs,
                 ) -> None:
        super().__init__(maxiter, disp, rhobeg, tol, options, **kwargs)
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
        # Remove ignored parameters to suppress the warning of scipy.optimize.minimize
        if self.is_bounds_ignored:
            bounds = None
        if self.is_gradient_ignored:
            jac = None

        if self.is_gradient_supported and jac is None and self._max_evals_grouped > 1:
            if "eps" in self._options:
                epsilon = self._options["eps"]
            else:
                epsilon = (
                    1e-8 if self._method in {"l-bfgs-b", "tnc"} else np.sqrt(
                        np.finfo(float).eps)
                )
            jac = Optimizer.wrap_function(
                Optimizer.gradient_num_diff,
                (fun, epsilon, self._max_evals_grouped)
            )

        # Workaround for L_BFGS_B because it does not accept np.ndarray.
        # See https://github.com/Qiskit/qiskit-terra/pull/6373.
        if jac is not None and self._method == "l-bfgs-b":
            jac = self._wrap_gradient(jac)

        # Starting in scipy 1.9.0 maxiter is deprecated and maxfun (added in 1.5.0)
        # should be used instead
        swapped_deprecated_args = False
        if self._method == "tnc" and "maxiter" in self._options:
            swapped_deprecated_args = True
            self._options["maxfun"] = self._options.pop("maxiter")

        raw_result = minimize(
            fun=fun,
            x0=x0,
            method=self._method,
            jac=jac,
            bounds=bounds,
            options=self._options,
            callback=self.callback,
            **self._kwargs,
        )
        if swapped_deprecated_args:
            self._options["maxiter"] = self._options.pop("maxfun")

        result = OptimizerResult()
        result.x = raw_result.x
        result.fun = raw_result.fun
        result.nfev = raw_result.nfev
        result.njev = raw_result.get("njev", None)
        result.nit = raw_result.get("nit", None)
        return result

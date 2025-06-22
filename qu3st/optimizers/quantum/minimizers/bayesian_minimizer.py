from qiskit_algorithms.optimizers import OptimizerResult
from typing import Callable

from bayes_opt import *
from bayes_opt.event import DEFAULT_EVENTS
from bayes_opt.target_space import TargetSpace

from .minimizer import Minimizer
import numpy as np
from bayes_opt.event import Events
import logging

logger = logging.getLogger(__name__)


class BayesianMinimizer(BayesianOptimization, Minimizer):

    def __init__(self,
                 pbounds,
                 constraint=None,
                 random_state=None,
                 verbose=0,
                 bounds_transformer=None,
                 allow_duplicate_points=True,
                 callback: Callable | None = None,
                 maxiter=1,
                 reload: bool = False,
                 p: float | None = None,
                 fun: str | None = None,
                 **kwargs):
        super().__init__(lambda *a, **kwa: 0,
                         pbounds,
                         constraint=constraint,
                         random_state=random_state,
                         verbose=verbose,
                         bounds_transformer=bounds_transformer,
                         allow_duplicate_points=allow_duplicate_points)
        self.random_state = random_state
        self.bounds_transformer = bounds_transformer
        self.callback = callback
        self.pbounds = pbounds
        self.maxiter = maxiter
        self.x = None
        self.reload = reload
        self.acquisition_function = None
        if p is not None and fun is not None:
            if fun == "ucb":
                self.acquisition_function = UtilityFunction(kind=fun, kappa=p)
            elif fun == "ei":
                self.acquisition_function = UtilityFunction(kind=fun, xi=p) # type: ignore
            elif fun == "poi":
                self.acquisition_function = UtilityFunction(kind=fun, xi=p) # type: ignore
            else:
                raise ValueError("Invalid string fro fun")

    def set_callback(self, callback: Callable):
        self.callback = callback

    def _get_wrapped_function(self, f):
        def wf(**kwargs):
            params = np.array([a for _, a in kwargs.items()])
            return -f(params)

        return wf

    def minimize(self, fun, x0, precomputed_steps=None, **kwargs):

        wf = self._get_wrapped_function(fun)

        if self.constraint is None:
            # Data structure containing the function to be optimized, the
            # bounds of its domain, and a record of the evaluations we have
            # done so far
            self._space = TargetSpace(
                wf, self.pbounds, random_state=self._random_state,
                allow_duplicate_points=self._allow_duplicate_points)
            self.is_constrained = False
        else:
            constraint_ = ConstraintModel(
                self.constraint.fun,
                self.constraint.lb,
                self.constraint.ub,
                random_state=self._random_state
            )
            self._space = TargetSpace(
                wf,
                self.pbounds,
                constraint=constraint_,
                random_state=self._random_state,
                allow_duplicate_points=self._allow_duplicate_points
            )
            self.is_constrained = True

        if self._bounds_transformer:
            try:
                self._bounds_transformer.initialize(self._space)
            except (AttributeError, TypeError):
                raise TypeError('The transformer must be an instance of '
                                'DomainTransformer')

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

        iteration = 0
        if precomputed_steps is None or len(precomputed_steps) == 0:
            initial_point = {f'x{i}': x0[i] for i in range(len(x0))}
            self.probe(params=initial_point, lazy=True)
        else:
            iteration = len(precomputed_steps)
            for tup in precomputed_steps:
                prm = np.array(tup["parameters"][0])
                if "ev" in tup.keys():
                    self.register(params=prm, target=tup["ev"])
                else:
                    self.probe(params=prm, lazy=True)
                    logger.info(f"Start from unfinished iteration {iteration}.")
                    # redo the last one
                    iteration -= 1

            logger.info(f"Registerd {iteration} points.")
        return self.maximize(n_iter=self.maxiter, iteration=iteration, **kwargs)

    def maximize(self,
                 init_points=0,
                 n_iter=25,
                 acq=None,
                 kappa=None,
                 kappa_decay=None,
                 kappa_decay_delay=None,
                 xi=None,
                 iteration=None,
                 **gp_params):
        r"""
        Maximize the given function over the target space.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        acq:
            Deprecated, unused and slated for deletion.

        kappa:
            Deprecated, unused and slated for deletion.

        kappa_decay:
            Deprecated, unused and slated for deletion.

        kappa_decay_delay:
            Deprecated, unused and slated for deletion.

        xi:
            Deprecated, unused and slated for deletion.

        \*\*gp_params:
            Deprecated, unused and slated for deletion.
        """

        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)

        old_params_used = any([param is not None for param in
                               [acq, kappa, kappa_decay, kappa_decay_delay,
                                xi]])
        if old_params_used or gp_params:
            raise Exception(
                '\nPassing acquisition function parameters or gaussian process parameters to maximize'
                '\nis no longer supported. Instead,please use the "set_gp_params" method to set'
                '\n the gp params, and pass an instance of bayes_opt.util.UtilityFunction'
                '\n using the acquisition_function argument\n')

        if self.acquisition_function is None:
            util = UtilityFunction(kind='ucb',
                                   kappa=2.576,
                                   xi=0,
                                   kappa_decay=1,
                                   kappa_decay_delay=0)
        else:
            util = self.acquisition_function

        if iteration is None:
            iteration = 0

        opt_res = OptimizerResult()
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1
            self.probe(x_probe, lazy=False)

            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

            if self.callback is not None and callable(self.callback):
                self.callback(x_probe)

            opt_res.x = np.array(x_probe)

        self.dispatch(Events.OPTIMIZATION_END)
        return opt_res

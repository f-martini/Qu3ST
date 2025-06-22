import numpy as np
from pathlib import Path
from docplex.mp.model import Model
from .objective_function import set_objective_function
from .balance_constraints import set_balance_constraints
from .link_constraints import set_link_constraints
from .collateral_constraint import set_collateral_constraints
from qu3st.optimizers.result import OptResult
import logging
logger = logging.getLogger(__name__)


class NullHandler(logging.Handler):
    def emit(self, record):
        pass


class CPLEXSolver:
    """
    Implements the NTS-problem solver leveraging cplex python API.
    """

    def __init__(self,
                 instance,
                 verbose=True,
                 collateral=True,
                 **kwargs):
        self.instance = instance
        self.verbose = verbose
        self.opt_args = kwargs
        self.collateral = collateral
        (self.model, self.t_vars, self.cb_inds, self.spl_vars,
         self.spl_inds) = self.get_model()
        self.set_constraints()
        self.model.parameters.mip.pool.intensity = 4 # type: ignore
        self.model.parameters.mip.tolerances.absmipgap = 10 ** -13 # type: ignore
        self.model.parameters.mip.tolerances.mipgap = 10 ** -13 # type: ignore
        self.model.parameters.simplex.tolerances.optimality = 10 ** -9 # type: ignore

    def get_model(self):
        model = Model(name='NTSP')
        n_t = len(self.instance.t_cash_amounts())
        n_spl = len(self.instance.spl_sp_receivers())
        n_cb = len(self.instance.cb_currencies())
        # add a boolean variables for each transaction
        t_vars = []
        for i in range(n_t):
            t_i = model.binary_var(name=f't_{i}')
            t_vars.append(t_i)
        # add a integer variable and a boolean slack variable for each SPL
        spl_vars = []
        spl_inds = []
        for i in range(n_spl):
            if self.collateral:
                spl_i = model.integer_var(lb=0, name=f'spl_{i}')
            else:
                spl_i = model.integer_var(lb=0, ub=0, name=f'spl_{i}')

            spl_ind = model.binary_var(name=f'spl_ind_{i}')
            spl_vars.append(spl_i)
            spl_inds.append(spl_ind)
        # add a slack variable for each CB
        cb_inds = []
        for i in range(n_cb):
            cb_i = model.binary_var(name=f'cb_{i}')
            cb_inds.append(cb_i)

        return model, t_vars, cb_inds, spl_vars, spl_inds

    def set_constraints(self):
        set_objective_function(
            instance=self.instance,
            model=self.model,
            t_vars=self.t_vars,
            **self.opt_args,
        )
        set_balance_constraints(
            instance=self.instance,
            model=self.model,
            t_vars=self.t_vars,
            spl_vars=self.spl_vars,
        )
        set_link_constraints(
            instance=self.instance,
            model=self.model,
            t_vars=self.t_vars,
        )
        set_collateral_constraints(
            instance=self.instance,
            model=self.model,
            t_vars=self.t_vars,
            cb_inds=self.cb_inds,
            spl_vars=self.spl_vars,
            spl_inds=self.spl_inds,
        )

    def solve(self, res_path: str | None = None):
        log_output = logger if self.verbose is True else None
        solution = self.model.solve(log_output=log_output)
        opt_res = OptResult(
                model=self,
                solver="classical",
                transactions=[int(np.round(solution.get_value(var))) for \
                              var in self.t_vars],
                collateral=[np.round(solution.get_value(var)) for var in
                            self.spl_vars],
                cb_indicators=[np.round(solution.get_value(var)) for var in
                               self.cb_inds],
                spl_indicators=[np.round(solution.get_value(var)) for var in
                                self.spl_inds],
                evaluation=solution.objective_value,
                evaluation_call=solution.solve_details.nb_iterations,
                runtime=solution.solve_details.time,
                custom_params={
                    "slack": [np.round(solution.get_value(var)) for var in
                              self.cb_inds],
                    "details": self.model.solve_details
                })
        if res_path is not None:
            opt_res.save(Path(res_path))
        return opt_res

    def get_fixed_solution_model(self, transactions, epsilon=10 ** -5):
        fs_model = self.model.copy(new_name="fixed_solution")
        for n, t_var in enumerate(self.t_vars):
            fs_t_var = fs_model.get_var_by_name(t_var.name)
            fs_t_var.lb = int(transactions[n]) - epsilon # type: ignore
            fs_t_var.ub = int(transactions[n]) + epsilon # type: ignore
        return fs_model

    def is_valid(self, transactions):
        fs_model = self.get_fixed_solution_model(transactions)
        original_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)  # Suppress output
        null_handler = NullHandler()
        logging.getLogger().addHandler(null_handler)

        fs_model.parameters.mip.pool.intensity = 4 # type: ignore
        fs_model.parameters.mip.tolerances.absmipgap = 10 ** -13 # type: ignore
        fs_model.parameters.mip.tolerances.mipgap = 10 ** -13 # type: ignore
        fs_model.parameters.simplex.tolerances.optimality = 10 ** -9 # type: ignore
        solution = fs_model.solve(log_output=False)

        logging.getLogger().removeHandler(null_handler)
        logging.getLogger().setLevel(original_level)

        return solution is not None and solution.objective_value is not None

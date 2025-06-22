from qu3st.ntsp.instance import Instance
from .cplex.cplex_solver import CPLEXSolver
from ..optimizer import Optimizer


class ClassicalOptimizer(Optimizer):

    def __init__(self,
                 instance: Instance,
                 opt_mode: str = "CPLEX",
                 opt_options: dict | None = None,
                 **kwargs):
        super().__init__(instance)
        self.opt_mode = opt_mode
        if opt_options is None:
            self.opt_options = {}
        else:
            self.opt_options = opt_options

    def solve(self, res_path: str | None = None, **kwargs):
        # initialize optimizer and solve NTS
        model = self.get_model(**kwargs)
        return model.solve(res_path)

    def get_model(self, **kwargs):
        if self.opt_mode == "CPLEX":
            return CPLEXSolver(self.instance, **self.opt_options, **kwargs)
        else:
            raise ValueError(f"{self.opt_mode} is not a valid optimization "
                             f"method.")

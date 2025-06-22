from abc import ABC, abstractmethod
from qu3st.ntsp.instance import Instance
from qu3st.optimizers.result import OptResult


class Optimizer(ABC):

    def __init__(self,
                 instance: Instance,
                 **kwargs):
        """
        Abstract class for NTS problem optimizers. The constructor takes as
        input the problem configuration.

        Args:
            instance: NTS problem instance
        """
        self.instance = instance

    @abstractmethod
    def solve(self) -> OptResult:
        """
        Solve NTS problem instance.
        """
        pass

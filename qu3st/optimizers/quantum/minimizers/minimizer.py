from abc import ABC, abstractmethod

class Minimizer(ABC):

    @abstractmethod
    def minimize(self, **kwargs):
        pass

    @abstractmethod
    def set_callback(self, **kwargs):
        pass

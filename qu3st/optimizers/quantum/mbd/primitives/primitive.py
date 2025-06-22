from abc import ABC, abstractmethod


class Primitive(ABC):

    def __init__(self):
        return

    @abstractmethod
    def run(self, *kwargs):
        pass

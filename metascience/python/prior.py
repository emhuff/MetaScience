from abc import ABCMeta, abstractmethod
import numpy as np

class prior(metaclass=ABCMeta):
    def _init__(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def draw(self):
        pass

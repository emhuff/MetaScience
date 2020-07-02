from abc import ABCMeta, abstractmethod
import numpy as np

class Cosmology(metaclass = ABCMeta):
    def __init__(self,complexity):
        pass

    @abstractmethod
    def generate_model_data_vector(self,):
        pass

    @abstractmethod
    def get_parameter_set(self):
        pass

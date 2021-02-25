from abc import ABCMeta, abstractmethod
import numpy as np

class Parameter:
    '''
    container for parameter values used in inference.
    'value' can be scalar or array-zeros_like
    'label' is a string that must include one of 'cosmology', 'nuisance', or 'systematics'

    '''
    def __init__(self,name = None, value = None, description = None, label = None):
        self.name = name
        self.value = value
        self.description = description
        assert ('systematics' in label) or ('nuisance' in label) or ('cosmology' in label)
        self.label = label

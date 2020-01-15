from abc import ABCMeta, abstractmethod

class ExperimentInterpreter(metaclass=ABCMeta):
    def __init__(self):
        pass

    @property
    def kind():
        '''
        What kind of experiment is being interpreted?
        '''
        raise NotImplementedError

    @abstractmethod
    def evaluate_posterior(parameters):
        '''
        What can we infer about the world from the provided experiment?
        Takes a cosmology, returns a log-posterior or similar.
        '''
        raise NotImplementedError
    
    @abstractmethod
    def fit_model():
        '''
        Fit a model. Generate a posterior.
        '''
        pass

        @abstractmethod
    def elaborate_systematics():
        '''
        Fit a model. Generate a posterior.
        '''
        pass

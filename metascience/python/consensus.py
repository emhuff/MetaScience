from abc import ABCMeta, abstractmethod

class Consensus(metaclass=ABCMeta):
    def __init__(self, interpretations = None):
        '''
        Expects a list-like object containing ExperimentInterpreter-like objects.
        Each should have an 'evaluate_posterior' method.
        

        '''
        pass

    @abstractmethhod
    def tension_metric():
        '''
        Should be able to take a combination of parameter posteriors, decide if there's tension,
         and return its decision in some kind of structured object that can be understood by 'render_judgment'
        '''
        raise NotImplementedError()

    @abstractmethod
    def render_judgment():
        '''
        Given the combination of tension and posteriors, choose to either:
          - update the cosmology, or 
          - tell one of the provided ExperimentInterpreter objects to update its nuisance parameter settings.
        
        Note: We may want to have this do more than just call the "update_systematics" method.
        '''
        raise NotImplementedError()

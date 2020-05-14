from abc import ABCMeta, abstractmethod

class Consensus(metaclass=ABCMeta):
    def __init__(self, interpretations = None):
        '''
        Expects a list-like object containing ExperimentInterpreter-like objects.
        Each should have an 'evaluate_posterior' method.


        '''
        pass

    @abstractmethhod
    def tension_metric(self):
        '''
        Should be able to take a combination of parameter posteriors, decide if
        there's tension, and return its decision in some kind of structured
        object that can be understood by 'render_judgment'
        '''
        raise NotImplementedError()

    @abstractmethod
    def _update_parameters(self):
        '''
        For this consensus, combine the results of the provided interpretation
         modules to get a best estimate of the *cosmological* parameters.
        '''
        pass

    @abstractmethod
    def render_judgment(self):
        '''
        Given the combination of tension and posteriors, choose to either:
          - update the cosmology, or
          - tell one of the provided ExperimentInterpreter objects to update its
          nuisance parameter settings.

        Note: We may want to have this do more than just call the
        "update_systematics" method.

        Call self._update_parameters()
        '''
        raise NotImplementedError()

class SensibleDefaultsConsensus(Consensus):
    def tension_metric(self):
        '''
        Define what you think the default tension metric should be
        '''
        pass

    def _update_parameters(self,judgments):
        # Mutiply together the posterios of the experiments that are *not* being directed to update their sytematics modules.
        # Store the resulting posterior in self.consensus_cosmological_parameters

    def render_judgment(self):
        pass

class SeminarConsensus(SensibleDefaultsConsensus):
    '''
    '''
    def __init__(self, interprations = None):
        super().__init__()

    def tension_metric(self):
        '''
        Count the maximum number of 'sigma'
         among the projected **cosmological** parameter contours
        '''

        def _convert_results_to_errorbars():

            return best_fit_cosmological_parameters, best_fit_cosmological_parameter_covariance



        pass

    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        pass


class AlwaysBetOnMeConsensus(Consensus):
    '''
    Define tension with respect to 0th interpretation, which never updates
    their model if there is a tension and all others do. Opposite of
    NeverBetOnThemConsensus if there are only 2 interpreters.
    '''
    def __init__(self, interpretations):
        super().__init__()
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = [False]*len(interpretations) # is this what we want?
        self.number_of_interpreters = len(interpretations)
        self.is_tension
        self.tm


    def tension_metric(self):
        '''
        Count the maximum number of 'sigma'
         among the projected **cosmological** parameter contours
        '''

        self.tm[0] = 0 #tension metric between others and the 0th interpreter.

        # for each experiment get tension with the "chosen one."
        for this_interp in self.interpretations[1:]:

            # difference between parameters inferred
            diff_vec = self.interpretations[0].best_fit_cosmological_parameters -
                self.interpretations[i].best_fit_cosmological_parameters

            # combination of covariance of parameters inferred
            joint_sum_cov = (self.interpretations[0].best_fit_cosmological_parameter_covariance +
                self.interpretations[i].best_fit_cosmological_parameter_covariance)

            # chisq difference in matrix form
            self.tm[i+1] = np.matmul(np.matmul(np.transpose(diff_vec), np.inv(joint_sum_cov)), diff_vec)
        pass

        if any(self.tm) > 1:
            self.is_tension=True

    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # interpreter 1 never changes their mode.
        # instruct both other interpreters to change systematics model
        # cosmology model remains fixed

        if self.is_tension == True:
            self.systematics_judgment[0] = 'you are cool' # or False?
            for i in range(self.number_of_interpreters-1):
                self.systematics_judgment[i+1] = 'you suck' # or True?
        pass

class NeverBetOnThemConsensus(Consensus):
    '''
    Define tension with respect to 0th interpretation, which always updates
    its systematics model if there is a tension, and the others never do.
    Opposite of AlwaysBetOnMeConsensus if there are only 2 interpreters.
    '''
    def __init__(self, interprations):
        super().__init__()
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = [False]*len(interpretations) # is this what we want?
        self.number_of_interpreters = len(interpretations)
        self.is_tension
        self.tm

    def tension_metric(self):
        '''
        Count the maximum number of 'sigma'
         among the projected **cosmological** parameter contours
        '''

        # Can this pick a "default" metric, e.g. the chisq in AlwaysBetOnMeConsensus?

        pass

    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # only 1 interpreter changes their systematics model.
        # instruct both other interpreters to stay fixed
        # cosmology model remains fixed

        if self.is_tension == True:
            self.systematics_judgment[0] = True
            for i in range(self.number_of_interpreters-1):
                self.systematics_judgment[i+1] = False

        pass

class EveryoneIsWrongConsensus(Consensus):
    '''
    '''
    def __init__(self, interprations):
        super().__init__()
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = [False]*len(interpretations) # is this what we want?
        self.number_of_interpreters = len(interpretations)
        self.is_tension
        self.tm

    def tension_metric(self):
        '''
        Count the maximum number of 'sigma'
         among the projected **cosmological** parameter contours
        '''

        pass

    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # all interpeters change their systematics model.
        # cosmology model remains fixed

        if self.is_tension == True:
            for i in range(self.number_of_interpreters):
                self.systematics_judgment[i] = True

        pass

class ShiftThatParadigmConsensus(Consensus):
    '''
    '''
    def __init__(self, interprations):
        super().__init__()
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = [False]*len(interpretations) # is this what we want?
        self.number_of_interpreters = len(interpretations)
        self.is_tension
        self.tm

    def tension_metric(self):
        '''
        Count the maximum number of 'sigma'
         among the projected **cosmological** parameter contours
        '''

        pass

    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.
        # see metric from above, decide if tension exists
        # all interpreters change their cosmologies

        if self.is_tension == True:
            for i in range(self.number_of_interpreters):
                self.cosmology_judgment[i] = True

        pass

class UnderestimatedErrorConsensus(Consensus):
    '''
    '''
    def __init__(self, interprations):
        super().__init__()

    def tension_metric(self):
        '''
        Count the maximum number of 'sigma'
         among the projected **cosmological** parameter contours
        '''

        pass

    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        # Based on this result, issue instructions.
        # see metric from above, decide if tension exists
        # All interpeters increase their errorbars

        # Q: should we make this one a me vs you as well?

        pass

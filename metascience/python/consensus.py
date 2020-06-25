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
        self.tm[0] = 0 #tension metric between others and the 0th interpreter.
        # for each experiment get tension with the "chosen one."
        for i, this_interp in enumerate(self.interpretations[1:]):

            # difference between parameters inferred
            diff_vec = self.interpretations[0].best_fit_cosmological_parameters - self.interpretations[i].best_fit_cosmological_parameters

            # combination of covariance of parameters inferred
            joint_sum_cov = (self.interpretations[0].best_fit_cosmological_parameter_covariance + self.interpretations[i].best_fit_cosmological_parameter_covariance)

            # chisq difference in matrix form
            self.tm[i+1] = np.matmul(np.matmul(np.transpose(diff_vec), np.inv(joint_sum_cov)), diff_vec)

        if any(self.tm) > 1:
            self.is_tension=True


    def _update_parameters(self,judgments):
            '''
            For this consensus, combine the results of the provided interpretation
             modules to get a best estimate of the *cosmological* parameters.
            '''
            chi2vec =self.interprations[i].chi2 for i in range(self.number_of_interpreters)]
            ind = np.where([chi2 ==   np.min(chi2vec) for chi2 in chi2vec])[0]

            self.consensus_cosmological_parameters = interpretations[ind].best_fit_cosmological_parameters
            self.consensus_parameter_covariance = interpretations[ind].best_fit_cosmological_parameter_covariance

    def render_judgment(self):

        if self.is_tension:
            for i, this_interp in enumerate(self.interpretations):

                if this_interp.chi2 > 3: # this is a totally arbitrary choice of number for now
                    self.systematics_judgment[i] = True

            if all(self.interpretations.chi2) < 3:
                [self.cosmology_judgment[i] = True for i in range(self.number_of_interpreters)]


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


class AlwaysBetOnMeConsensus(SensibleDefaultsConsensus):
    '''
    Define tension with respect to 0th interpretation, which never updates
    their model if there is a tension and all others do. Opposite of
    NeverBetOnThemConsensus if there are only 2 interpreters.
    '''
    def __init__(self, interpretations):
        super().__init__()
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = [False]*len(interpretations)
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm


    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # interpreter 1 never changes their mode.
        # instruct both other interpreters to change systematics model
        # cosmology model remains fixed

        # call the tension metric from the default
        tension_metric() #is this correct???

        if self.is_tension == True:
            self.systematics_judgment[0] = False
            for i in range(self.number_of_interpreters-1):
                self.systematics_judgment[i+1] = True

    def _update_parameters(self):
        '''
        For this consensus, combine the results of the provided interpretation
         modules to get a best estimate of the *cosmological* parameters.
        '''
        self.consensus_cosmological_parameters = interpretations[0].best_fit_cosmological_parameters
        self.consensus_parameter_covariance = interpretations[0].best_fit_cosmological_parameter_covariance


class NeverBetOnThemConsensus(SensibleDefaultsConsensus):
    '''
    Define tension with respect to 0th interpretation, which always updates
    its systematics model if there is a tension, and the others never do.
    Opposite of AlwaysBetOnMeConsensus if there are only 2 interpreters.
    '''
    def __init__(self, interprations):
        super().__init__()
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = [False]*len(interpretations)
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm


    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # only 1 interpreter changes their systematics model.
        # instruct both other interpreters to stay fixed
        # cosmology model remains fixed

        # call the tension metric from the default
        tension_metric() #is this correct???

        if self.is_tension == True:
            self.systematics_judgment[0] = True
            for i in range(self.number_of_interpreters-1):
                self.systematics_judgment[i+1] = False

    def _update_parameters(self):
        '''
        For this consensus, combine the results of the provided interpretation
         modules to get a best estimate of the *cosmological* parameters.
        '''
        self.consensus_cosmological_parameters = np.mean(interpretations[1:].best_fit_cosmological_parameters)
        self.consensus_parameter_covariance = np.matrix.mean(interpretations[1:].best_fit_cosmological_parameter_covariance)

        pass

class EveryoneIsWrongConsensus(SensibleDefaultsConsensus):
    '''
    '''
    def __init__(self, interprations):
        super().__init__()
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = [False]*len(interpretations)
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm



    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # all interpeters change their systematics model.
        # cosmology model remains fixed

        self.tension_metric()

        if self.is_tension == True:
            for i in range(self.number_of_interpreters):
                self.systematics_judgment[i] = True

        def _update_parameters(self):
            '''
            For this consensus, combine the results of the provided interpretation
            modules to get a best estimate of the *cosmological* parameters.
            '''

#             do not return any consensus or best fit here - everyone is terrible

        pass

class ShiftThatParadigmConsensus(SensibleDefaultsConsensus):
    '''
    '''
    def __init__(self, interprations):
        super().__init__()
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = [False]*len(interpretations)
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm

    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.
        # see metric from above, decide if tension exists
        # all interpreters change their cosmologies

        tension_metric()
        if self.is_tension == True:
            for i in range(self.number_of_interpreters):
                self.cosmology_judgment[i] = True

         # here we need to fix a new base model and communicate that to all the interpreters

        pass

class UnderestimatedErrorConsensus(SensibleDefaultsConsensus):
    '''
    '''
    def __init__(self, interprations):
        super().__init__()

    def render_judgment(self):

        tension_metric()

        # Measure the tension among the provided interpretation objects
        # Based on this result, issue instructions.
        # see metric from above, decide if tension exists
        # All interpeters increase their errorbars

        # Q: should we make this one a me vs you as well?

        if self.is_tension == True:
            print('code this')
        pass

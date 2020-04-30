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
    def render_judgment(self):
        '''
        Given the combination of tension and posteriors, choose to either:
          - update the cosmology, or
          - tell one of the provided ExperimentInterpreter objects to update its
          nuisance parameter settings.

        Note: We may want to have this do more than just call the
        "update_systematics" method.
        '''
        raise NotImplementedError()


class SeminarConsensus(Consensus):
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
    This consensus module only really makes sense  for > 2 interpeters, otherwise it reduces to the same as the NeverBetOnThemConsensus
    '''
    def __init__(self, interpretations):
        super().__init__()
        self.interpretations = interpretations
        self.systematics_judgement = [False]*len(interpretations)
        self.cosmology_judgement = [False]*len(interpretations) # is this what we want?
        self.number_of_interpreters = len(interpretations)
        self.tension
        self.tm


    def tension_metric(self):
        '''
        Count the maximum number of 'sigma'
         among the projected **cosmological** parameter contours
        '''

        self.tm[0] = 0 #tension metric between others and the 0th interpreter.

        # for each experiment get tension with the "chosen one."
        for this_interp in self.interpratations[1:]:

            # difference between parameters inferred
            diff_vec = interpretations[0].best_fit_cosmological_parameters -
                interpretations[i].best_fit_cosmological_parameters

            # combination of covariance of parameters inferred
            joint_sum_cov = (interpretations[0].best_fit_cosmological_parameter_covariance +
                interpretations[i].best_fit_cosmological_parameter_covariance)

            # chisq difference in matrix form
            self.tm[i+1] = np.matmul(np.matmul(np.transpose(diff_vec), np.inv(joint_sum_cov)), diff_vec)
        pass

        if any(self.tm) > 1:
            self.tension=True

    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # interpreter 1 never changes their mode.
        # instruct both other interpreters to change systematics model
        # cosmology model remains fixed

        if self.tension == True:
            interpretations[0].systematics_judgement= 'you are cool'
            for i in range(self.number_of_interpreters-1):
                interpretations[i+1].systematics_judgement= 'you suck'
        pass

class NeverBetOnThemConsensus(Consensus):
    '''
    This consensus module only really makes sense  for > 2 interpeters, otherwise it reduces to the same as the AlwaysBetOnMeConsensus
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
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # only 1 interpreter changes their systematics model.
        # instruct both other interpreters to stay fixed
        # cosmology model remains fixed
        pass

class EveryoneIsWrongConsensus(Consensus):
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
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # all interpeters change their systematics model.
        # cosmology model remains fixed

        pass

class ShiftThatParadigmConsensus(Consensus):
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
        #  Based on this result, issue instructions.
        # see metric from above, decide if tension exists
        # all interpreters change their cosmologies


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

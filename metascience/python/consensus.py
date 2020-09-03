from abc import ABCMeta, abstractmethod
import numpy as np

class Consensus(metaclass=ABCMeta):
    def __init__(self, interpretations = None):
        '''
        Expects a list-like object containing ExperimentInterpreter-like objects.
        Each should have an 'evaluate_posterior' method.


        '''
        pass

    @abstractmethod
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


    def __init__(self, interpretations = None, chi2_dof_threshold = 1.25):
        super().__init__(interpretations = None )
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = False # [False]*len(interpretations)
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm = np.zeros(len(interpretations))
        self.chi2_dof_threshold = chi2_dof_threshold


    def tension_metric(self):
        '''
        Define what you think the default tension metric should be
        '''
        self.tm[0] = 0 #tension metric between others and the 0th interpreter.
        # for each experiment get tension with the "chosen one."
        for i, this_interp in enumerate(self.interpretations[1:]):

            # difference between parameters inferred
            diff_vec = self.interpretations[0].best_fit_cosmological_parameters - this_interp.best_fit_cosmological_parameters
            # combination of covariance of parameters inferred
            joint_sum_cov = (self.interpretations[0].best_fit_cosmological_parameter_covariance + this_interp.best_fit_cosmological_parameter_covariance)

            # chisq difference in matrix form

            self.tm[i+1] = np.matmul(np.matmul(np.transpose(diff_vec), np.linalg.inv(joint_sum_cov)), diff_vec)
        if np.sum(self.tm > 1.) > 0:
            self.is_tension=True


    def _update_parameters(self):
            '''
            For this consensus, combine the results of the provided interpretation
             modules to get a best estimate of the *cosmological* parameters.
            '''
            chi2vec = [self.interpretations[i].chi2 for i in range(self.number_of_interpreters)]
            #ind = np.where([chi2 ==   np.min(chi2vec) for chi2 in chi2vec])[0][0]
            ind = np.argmin(chi2vec)

            self.consensus_cosmological_parameters = self.interpretations[ind].best_fit_cosmological_parameters
            self.consensus_parameter_covariance = self.interpretations[ind].best_fit_cosmological_parameter_covariance

    def render_judgment(self):
        '''
        If there is a tension and the reduced chi2 values are high, update systematics.
        If there is a tension and they are low, change the cosmology model.
        '''
        if self.is_tension:

            chi2_list = np.array([thing.chi2*1./thing.measured_data_vector.size for thing in self.interpretations])

#            this_interp.chi2*1./this_interp.measured_data_vector.size

            for i, this_interp in enumerate(self.interpretations):
                if chi2_list[i] >= self.chi2_dof_threshold: # this is a totally arbitrary choice of number for now
                    self.systematics_judgment[i] = True

            if all(chi2_list < self.chi2_dof_threshold):
                self.cosmology_judgment = True


        self._update_parameters()
                #for i in range(self.number_of_interpreters):
                #    self.cosmology_judgment[i] = True


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
        self.cosmology_judgment = False # [False]*len(interpretations)
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm = np.zeros(len(interpretations))


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
        self.cosmology_judgment = False #[False]*len(interpretations)
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm = np.zeros(len(interpretations))


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
        self.cosmology_judgment = False #[False]*len(interpretations)
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm = np.zeros(len(interpretations))



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
        self.cosmology_judgment = False #[False]*len(interpretations)
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm = np.zeros(len(interpretations))


    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.
        # see metric from above, decide if tension exists
        # all interpreters change their cosmologies

        tension_metric()
        if self.is_tension == True:
            self.cosmology_judgment = True

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

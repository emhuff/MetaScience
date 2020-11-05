from abc import ABCMeta, abstractmethod
import numpy as np

class Consensus(metaclass=ABCMeta):
    def __init__(self, interpretations = None):
        '''
        Expects a list-like object containing ExperimentInterpreter-like objects.
        Each should have an 'evaluate_posterior' method.
        '''
        self.name = None
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


class DefaultConsensus(Consensus):
    '''
    philosophy: update model to include more systematics until the paradigm
    changes
    '''

    def __init__(self, interpretations = None, chi2_dof_threshold = 1.25):
        super().__init__( interpretations = interpretations )
        self.name = 'Default Consensus'
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = False
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
             Pick values from interpreter with lowest chi2
            '''
            chi2vec = [self.interpretations[i].chi2 for i in range(self.number_of_interpreters)]
            #ind = np.where([chi2 ==   np.min(chi2vec) for chi2 in chi2vec])[0][0]
            ind = np.argmin(chi2vec)

            self.consensus_cosmological_parameters = self.interpretations[ind].best_fit_cosmological_parameters
            self.consensus_parameter_covariance = self.interpretations[ind].best_fit_cosmological_parameter_covariance

    def render_judgment(self):
        '''
        If the reduced chi2 values are high, update systematics, WHETHER OR NOT there is a tension.
        If there is a tension and they are low, change the cosmology model.
        '''

        chi2_list = np.array([thing.chi2*1./thing.measured_data_vector.size for thing in self.interpretations])

        for i, this_interp in enumerate(self.interpretations):
            if chi2_list[i] >= self.chi2_dof_threshold: # this is a totally arbitrary choice of number for now
                self.systematics_judgment[i] = True

        if self.is_tension:

            # for i, this_interp in enumerate(self.interpretations):
            #     if chi2_list[i] >= self.chi2_dof_threshold: # this is a totally arbitrary choice of number for now
            #         self.systematics_judgment[i] = True

            if all(chi2_list < self.chi2_dof_threshold):
                self.cosmology_judgment = True


        self._update_parameters()
                #for i in range(self.number_of_interpreters):
                #    self.cosmology_judgment[i] = True



class ImpatientConsensus(DefaultConsensus):
    '''
    same as default, but with patience parameter
    an instance of what used to be "sensible" (now deprecated)
    '''

    def __init__(self, interpretations = None, chi2_dof_threshold = 1.25, patience = 10):
        super().__init__(interpretations = interpretations)
        self.name = 'Impatient Consensus'
        self.chi2_dof_threshold = chi2_dof_threshold
        self.patience = patience

    def render_judgment(self, number_of_tries = 0):
        '''
        If the reduced chi2 values are high, update systematics, WHETHER OR NOT there is a tension.
        If there is a tension and they are low, change the cosmology model.
        If the fits are not all good, but we've been updating systematics for too long, change the cosmology.
        '''

        chi2_list = np.array([thing.chi2*1./thing.measured_data_vector.size for thing in self.interpretations])

        for i, this_interp in enumerate(self.interpretations):
            if chi2_list[i] >= self.chi2_dof_threshold: # this is a totally arbitrary choice of number for now
                self.systematics_judgment[i] = True

        if self.is_tension:
            if all(chi2_list < self.chi2_dof_threshold):
                self.cosmology_judgment = True

#        if (np.sum(chi2_list >= self.chi2_dof_threshold) > 0) and (number_of_tries > self.patience):
# if at least one interpreter has been told to update systematics, but patience has been reached
        if (np.sum(self.systematics_judgment) > 0) and (number_of_tries > self.patience):
            self.cosmology_judgment = True
            self.systematics_judgment = [False]*len(self.interpretations)


        self._update_parameters()
                #for i in range(self.number_of_interpreters):
                #    self.cosmology_judgment[i] = True


class TensionOnlyConsensus(ImpatientConsensus):
    """
    Same as ImpatientConsensus, but only update systematics if there IS a tension,
    and ignore case where there are bad fits but no tension.
    """

    def __init__(self, interpretations = None, chi2_dof_threshold = 1.25, patience = 10):
        super().__init__(interpretations = interpretations, chi2_dof_threshold = chi2_dof_threshold , patience = patience)
        self.name = 'TensionOnly Consensus'
        #pretty sure this is all you need to do.... and that code below could be updated accordingly... (BF)

    def render_judgment(self,number_of_tries=0):

        chi2_list = np.array([thing.chi2*1./thing.measured_data_vector.size for thing in self.interpretations])

        if self.is_tension:
            for i, this_interp in enumerate(self.interpretations):
                if chi2_list[i] >= self.chi2_dof_threshold: # this is a totally arbitrary choice of number for now
                    self.systematics_judgment[i] = True
            if all(chi2_list < self.chi2_dof_threshold):
                self.cosmology_judgment = True

        if (np.sum(self.systematics_judgment) > 0) and (number_of_tries > self.patience):
            self.cosmology_judgment = True
            self.systematics_judgment = [False]*len(self.interpretations)

        self._update_parameters()


class AlwaysBetOnMeConsensus(DefaultConsensus):
    '''
    Define tension with respect to 0th interpretation, which never updates
    their model if there is a tension and all others do. Opposite of
    NeverBetOnThemConsensus if there are only 2 interpreters.

    missing: doesn't update cosmology (older), and not impatient

    philosophy: we don't really ever change, we want others to change their
    models.
    '''
    def __init__(self, interpretations):
        super().__init__()
        self.name = 'AlwaysBetOnMe Consensus'
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = False
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm = np.zeros(len(interpretations))

    def _update_parameters(self):
        '''
        For this consensus, the best estimate of the *cosmological* parameters
        is provided by the 0th interpretation.
        '''
        self.consensus_cosmological_parameters = interpretations[0].best_fit_cosmological_parameters
        self.consensus_parameter_covariance = interpretations[0].best_fit_cosmological_parameter_covariance

    def render_judgment(self):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # interpreter 1 never changes their mode.
        # instruct both other interpreters to change systematics model
        # cosmology model remains fixed

        if self.is_tension == True:
            self.systematics_judgment[0] = False
            for i in range(self.number_of_interpreters-1):
                self.systematics_judgment[i+1] = True

        self._update_parameters()


class MostlyBetOnMeConsensus(ImpatientConsensus):
    '''
    Define tension with respect to 0th interpretation, which has a higher error
    tolerance so is less likely to update their systematics model. Otherwise
    it is like the Sensible Defaults decision rule.

    philosophy: softer than always bet on me(?); has a different chi-sq thresh
    for changing the cosmology
    '''
    def __init__(self, interpretations, tolerance = 2, chi2_dof_threshold = 1.25, patience = 10):
        super().__init__(interpretations = interpretations, chi2_dof_threshold = chi2_dof_threshold , patience = patience)
        self.name = 'MostlyBetOnMe Consensus'
        self.tolerance = tolerance


    def render_judgment(self, number_of_tries = 0):
        '''
        Measure the tension among the provided interpretation objects
        Based on this result, issue instructions.

        see metric from above, decide if tension exists
            interpreter 1 changes their model less often.
        instruct both other interpreters to change systematics model if fits bad
            cosmology model updates if there is a tension and fits are good
        '''

        chi2_list = np.array([thing.chi2*1./thing.measured_data_vector.size for thing in self.interpretations])

        for i, this_interp in enumerate(self.interpretations):
            if i == 0:
                if chi2_list[i] >= self.tolerance*self.chi2_dof_threshold:
                    self.systematics_judgment[i] = True
            else:
                if chi2_list[i] >= self.chi2_dof_threshold:
                    self.systematics_judgment[i] = True

        if self.is_tension:
            if all(chi2_list < self.chi2_dof_threshold):
                self.cosmology_judgment = True

# if at least one interpreter has been told to update systematics, but patience has been reached
        if (np.sum(self.systematics_judgment) > 0) and (number_of_tries > self.patience):
            self.cosmology_judgment = True
            self.systematics_judgment = [False]*len(self.interpretations)

        self._update_parameters()



class AlwaysBetOnThemConsensus(DefaultConsensus):
    '''
    Define tension with respect to 0th interpretation, which always updates
    its systematics model if there is a tension, and the others never do.
    Opposite of AlwaysBetOnMeConsensus if there are only 2 interpreters.

    missing: doesn't update cosmology (older), and not impatient

    philosophy: Impostor syndrome version; people who are very quick to revise
    their model (rare)
    '''
    def __init__(self, interprations):
        super().__init__()
        self.name = 'AlwaysBetOnThem Consensus'
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = False
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm = np.zeros(len(interpretations))


    def _update_parameters(self):
        '''
        For this consensus, combine the results of the provided interpretation
         modules to get a best estimate of the *cosmological* parameters.
        '''
        self.consensus_cosmological_parameters = np.mean(interpretations[1:].best_fit_cosmological_parameters)
        self.consensus_parameter_covariance = np.matrix.mean(interpretations[1:].best_fit_cosmological_parameter_covariance)

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

        self._update_parameters()


class MostlyBetOnThemConsensus(ImpatientConsensus):
    '''
    Corollary to MostlyBetOnMeConsensus

    All interpretations besides 0th have a higher error
    tolerance so are less likely to update their systematics model
    '''

    def __init__(self, interprations, tolerance = 2, chi2_dof_threshold = 1.25, patience = 10):
        super().__init__(interpretations = interpretations, chi2_dof_threshold = chi2_dof_threshold , patience = patience)
        self.name = 'MostlyBetOnThem Consensus'


    def _update_parameters(self):
        '''
        For this consensus, combine the results of the provided interpretation
         modules to get a best estimate of the *cosmological* parameters.
        '''
        self.consensus_cosmological_parameters = np.mean(interpretations[1:].best_fit_cosmological_parameters)
        self.consensus_parameter_covariance = np.matrix.mean(interpretations[1:].best_fit_cosmological_parameter_covariance)


    def render_judgment(self, number_of_tries = 0):
        '''
        Measure the tension among the provided interpretation objects
        Based on this result, issue instructions.

        see metric from above, decide if tension exists
            interpreter 1 changes their model more often.
        instruct both other interpreters to change systematics model if fits bad
            cosmology model updates if there is a tension and fits are good
        '''

        chi2_list = np.array([thing.chi2*1./thing.measured_data_vector.size for thing in self.interpretations])

        for i, this_interp in enumerate(self.interpretations):
            if i == 0:
                if chi2_list[i] >= self.chi2_dof_threshold:
                    self.systematics_judgment[i] = True
            else:
                if chi2_list[i] >= self.tolerance*self.chi2_dof_threshold:
                    self.systematics_judgment[i] = True

        if self.is_tension:
            if all(chi2_list < self.chi2_dof_threshold):
                self.cosmology_judgment = True
        if (np.sum(self.systematics_judgment) > 0) and (number_of_tries > self.patience):
            self.cosmology_judgment = True
            self.systematics_judgment = [False]*len(self.interpretations)

        self._update_parameters()


class EveryoneIsWrongConsensus(DefaultConsensus):
    '''
    need: good neutral evil

    philosophy: ...
    '''
    def __init__(self, interprations):
        super().__init__()
        self.name = 'EveryoneIsWrong Consensus'
        self.interpretations = interpretations
        self.systematics_judgment = [False]*len(interpretations)
        self.cosmology_judgment = False
        self.number_of_interpreters = len(interpretations)
        self.is_tension = False
        self.tm = np.zeros(len(interpretations))

    def _update_parameters(self):
        '''
        For this consensus, combine the results of the provided interpretation
        modules to get a best estimate of the *cosmological* parameters.
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

        self._update_parameters()


class ShiftThatParadigmConsensus(DefaultConsensus):
    '''
    If there is a tension, everyone updates their cosmology.
    Never tells interpreters to update systematics.
    '''
    def __init__(self, interpretations):
        super().__init__(interpretations=interpretations)
        self.name = 'ShiftThatParadigm Consensus'



    def render_judgment(self):
        # see metric from DefaultConsensus, decide if tension exists
        # all interpreters change their cosmologies

        if self.is_tension == True:
            self.cosmology_judgment = True

        self._update_parameters()


class UnderestimatedErrorConsensus(DefaultConsensus):
    '''
    TO DO: What is this?
    '''
    def __init__(self, interprations):
        super().__init__()
        self.name = 'UnderestimatedError Consensus'

    def render_judgment(self):


        # Measure the tension among the provided interpretation objects
        # Based on this result, issue instructions.
        # see metric from above, decide if tension exists
        # All interpeters increase their errorbars

        # Q: should we make this one a me vs you as well?

        if self.is_tension == True:
            print('code this')
        pass



class SeminarConsensus(DefaultConsensus):
    '''
    philosophy: ???
    '''
    def __init__(self, interprations = None):
        super().__init__()
        self.name = 'Seminar Consensus'

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

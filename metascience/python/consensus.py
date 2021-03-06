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
            nsample = 10000
            samples = np.random.multivariate_normal(mean=np.zeros(this_interp.best_fit_cosmological_parameters.size),cov=joint_sum_cov,size=nsample)
            logL = np.zeros(nsample)
            for j in range(nsample):
                this_diff = samples[j,:]
                logL[j] = np.matmul(np.matmul(np.transpose(this_diff), np.linalg.inv(joint_sum_cov)), this_diff)
            self.tm[i+1] = np.matmul(np.matmul(np.transpose(diff_vec), np.linalg.inv(joint_sum_cov)), diff_vec)
            tm_thresh = np.percentile(logL,99.9)#/100. # RH manually made the threshold small
            print(f"tension metric threshold: {tm_thresh}")
        if np.sum(self.tm > tm_thresh) > 0:
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


class AlwaysBetOnMeConsensus(ImpatientConsensus):
    '''
    Define tension with respect to 0th interpretation, which never updates
    their model if there is a tension and all others do. Opposite of
    NeverBetOnThemConsensus if there are only 2 interpreters.

    missing: doesn't update cosmology (older), and not impatient

    philosophy: we don't really ever change, we want others to change their
    models.
    '''
    def __init__(self, interpretations, chi2_dof_threshold = 1.25, patience = 10):
        super().__init__(interpretations = interpretations,chi2_dof_threshold = chi2_dof_threshold, patience = patience)
        self.name = 'AlwaysBetOnMe Consensus'


    def _update_parameters(self):
        '''
        For this consensus, the best estimate of the *cosmological* parameters
        is provided by the 0th interpretation.
        '''
        self.consensus_cosmological_parameters = self.interpretations[0].best_fit_cosmological_parameters
        self.consensus_parameter_covariance = self.interpretations[0].best_fit_cosmological_parameter_covariance

    def render_judgment(self,number_of_tries = 0):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # interpreter 1 never changes their mode.
        # instruct both other interpreters to change systematics model
        # cosmology model remains fixed

        chi2_list = np.array([thing.chi2*1./thing.measured_data_vector.size for thing in self.interpretations])

# this loop shows that all other need to change systematics
        for i in range(self.number_of_interpreters-1):
            if chi2_list[i+1] >= self.chi2_dof_threshold:
                self.systematics_judgment[i+1] = True
                print("bad fit for other people, updating systematics")

# if there is tension but we all have a good fit, change cosmology
        if self.is_tension:
            if all(chi2_list < self.chi2_dof_threshold):
                self.cosmology_judgment = True

# if I'm impatient change cosmology
        if (np.sum(self.systematics_judgment) > 0) and (number_of_tries > self.patience):
            self.cosmology_judgment = True
        #    self.systematics_judgment = [False]*len(self.interpretations)
            print(f"checking we did change the cosmo {self.cosmology_judgment}")

        self._update_parameters()


class MostlyBetOnMeConsensus(ImpatientConsensus):
    '''
    Define tension with respect to 0th interpretation, which has a higher error
    tolerance so is less likely to update their systematics model. Otherwise
    it is like the Sensible Defaults decision rule.

    philosophy: softer than always bet on me(?); has a different chin-sq thresh
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
                    print("bad fit, updating systematics")
            else:
                if chi2_list[i] >= self.chi2_dof_threshold:
                    self.systematics_judgment[i] = True
                    print("bad fit, updating systematics")

        if self.is_tension:
            if all(chi2_list < self.chi2_dof_threshold):
                self.cosmology_judgment = True

# if at least one interpreter has been told to update systematics, but patience has been reached
        if (np.sum(self.systematics_judgment) > 0) and (number_of_tries > self.patience):
            self.cosmology_judgment = True
            self.systematics_judgment = [False]*len(self.interpretations)

        self._update_parameters()



class AlwaysBetOnThemConsensus(ImpatientConsensus):
    '''
    Define tension with respect to 0th interpretation, which always updates
    its systematics model if there is a tension, and the others never do.
    Opposite of AlwaysBetOnMeConsensus if there are only 2 interpreters.

    missing: doesn't update cosmology (older), and not impatient

    philosophy: Impostor syndrome version; people who are very quick to revise
    their model (rare)
    '''
    def __init__(self, interpretations, chi2_dof_threshold = 1.25, patience = 10):
        super().__init__(interpretations = interpretations,chi2_dof_threshold = chi2_dof_threshold, patience = patience)
        self.name = 'AlwaysBetOnThem Consensus'


    def _update_parameters(self):
        '''
        For this consensus, combine the results of the provided interpretation
         modules to get a best estimate of the *cosmological* parameters.
        '''
        self.consensus_cosmological_parameters=0*self.interpretations[0].best_fit_cosmological_parameters
        self.consensus_parameter_covariance = 0*self.interpretations[0].best_fit_cosmological_parameter_covariance

        for i in range(self.number_of_interpreters-1):
            self.consensus_cosmological_parameters += self.interpretations[i+1].best_fit_cosmological_parameters
            self.consensus_parameter_covariance += self.interpretations[i+1].best_fit_cosmological_parameter_covariance

        self.consensus_cosmological_parameters/(self.number_of_interpreters-1)
        self.consensus_parameter_covariance/(self.number_of_interpreters-1)

    def render_judgment(self, number_of_tries = 0):
        # Measure the tension among the provided interpretation objects
        #  Based on this result, issue instructions.

        # see metric from above, decide if tension exists
        # only 1 interpreter changes their systematics model.
        # instruct both other interpreters to stay fixed
        # cosmology model remains fixed

        chi2_list = np.array([thing.chi2*1./thing.measured_data_vector.size for thing in self.interpretations])
# 0th interpreter needs to change systematics
        if chi2_list[0] >= self.chi2_dof_threshold:
            self.systematics_judgment[0] = True
            print("bad fit for me, updating systematics")

        if self.is_tension:
            if all(chi2_list < self.chi2_dof_threshold):
                self.cosmology_judgment = True


        if self.is_tension == True:
            self.systematics_judgment[0] = True
            for i in range(self.number_of_interpreters-1):
                self.systematics_judgment[i+1] = False

# if at least one interpreter has been told to update systematics, but patience has been reached
        if (np.sum(self.systematics_judgment) > 0) and (number_of_tries > self.patience):
            self.cosmology_judgment = True
            self.systematics_judgment = [False]*len(self.interpretations)

        self._update_parameters()


class MostlyBetOnThemConsensus(ImpatientConsensus):
    '''
    Corollary to MostlyBetOnMeConsensus

    All interpretations besides 0th have a higher error
    tolerance so are less likely to update their systematics model
    '''

    def __init__(self, interpretations, tolerance = 2, chi2_dof_threshold = 1.25, patience = 10):
        super().__init__(interpretations = interpretations, chi2_dof_threshold = chi2_dof_threshold , patience = patience)
        self.name = 'MostlyBetOnThem Consensus'
        self.tolerance = tolerance


    def _update_parameters(self):
        '''
        For this consensus, combine the results of the provided interpretation
         modules to get a best estimate of the *cosmological* parameters.
        '''
        ## TODO: Revisit once the cosmological parameters vectors are longer than 1.
        self.consensus_cosmological_parameters = np.mean([interp.best_fit_cosmological_parameters for interp in  self.interpretations[1:]])
        self.consensus_parameter_covariance = np.mean([interp.best_fit_cosmological_parameter_covariance for interp in  self.interpretations[1:]])


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


# class EveryoneIsWrongConsensus(ImpatientConsensus):
#     '''
#     need: good neutral evil
#
#     philosophy: ...
#     '''
#     def __init__(self, interprations):
#         super().__init__()
#         self.name = 'EveryoneIsWrong Consensus'
#         self.interpretations = interpretations
#         # self.systematics_judgment = [False]*len(interpretations)
#         # self.cosmology_judgment = False
#         # self.number_of_interpreters = len(interpretations)
#         # self.is_tension = False
#         # self.tm = np.zeros(len(interpretations))
#
#     def _update_parameters(self):
#         '''
#         For this consensus, combine the results of the provided interpretation
#         modules to get a best estimate of the *cosmological* parameters.
#         '''
#         pass
#
#     def render_judgment(self):
#         # Measure the tension among the provided interpretation objects
#         #  Based on this result, issue instructions.
#
#         # see metric from above, decide if tension exists
#         # all interpeters change their systematics model.
#         # cosmology model remains fixed
#
#         if self.is_tension == True:
#             for i in range(self.number_of_interpreters):
#                 self.systematics_judgment[i] = True
#
#         self._update_parameters()


class ShiftThatParadigmConsensus(ImpatientConsensus):
    '''
    If there's a bad fit (regardless of tension), update systematics.
    If there is a tension, everyone updates their cosmology.
    '''


    def __init__(self, interpretations, chi2_dof_threshold = 1.25, patience = None):
        super().__init__(interpretations = interpretations, chi2_dof_threshold = chi2_dof_threshold, patience = patience)
        self.name = 'ShiftThatParadigm Consensus'


    def render_judgment(self, number_of_tries = 0):
        # see metric from DefaultConsensus, decide if tension exists
        # all interpreters change their cosmologies

        chi2_list = np.array([thing.chi2*1./thing.measured_data_vector.size for thing in self.interpretations])

# this loop shows that all other need to change systematics
        for i in range(self.number_of_interpreters):
            if chi2_list[i] >= self.chi2_dof_threshold:
                self.systematics_judgment[i] = True
                print("bad fit, updating systematics")

        if self.is_tension == True:
            self.cosmology_judgment = True

        self._update_parameters()


class UnderestimatedErrorConsensus(DefaultConsensus):
    '''
    TO DO: Deveop a consensus model that only thinks tensions can be solved by
    increasing the magnitude of the errors.
    If there is a tension:
        1. Figure out how large the errors (of all interpreters?) need to be
        in order to solve the tension.
            which interpreters? for now, all of them with same fractional amount
        2. Determine if this increase is "reasonable" or too large
            increase errors by a maximum fraction, between 5 and 10?
        3. If the proposed error increase is too large, direct interpreters
        to update the cosmology.
    Subject to change... note that as written this consensus would never tell
    interpreters to update their systematics models...

    As a reminder: the scripts call consensus.tension_metric, then print out
    consensus.tm and consensus.is_tension, then call consensus.render_judgment.
    '''

    def __init__(self, interpretations, chi2_dof_threshold = 1.25, patience = None):
        super().__init__(interpretations = interpretations)
        self.name = 'UnderestimatedError Consensus'
        self.tm_thresh = np.zeros(len(interpretations))
        self.max_bias = 1 #maximum fraction of error increase allowed ## SHOuLD this be a kwarg??
        self.patience = patience

    def render_judgment(self, number_of_tries = None):
        '''
        If there is a tension as determined by tension_metric,
          iteratively increase all errors (interpretations.best_fit_cosmological_parameter_covariance ??)
          and recompute self.tm (call tension_metric within this function?)
          until self.is_tension = False.
        Then compare err_increase to max_err_increase (defined in init, or passed as argument to this function?)
          and if err_increase >= max_err_increase: cosmology_judgment = True,
          otherwise, report the required increase in some way...
          (We don't have a mechanism to tell interpreters to "increase errors by X factor" yet.)
        '''

        #We need some way to update the systematics judgment in case of bad fits even when there's no tension (like models based on ImpatientConsensus)
        chi2_list = np.array([thing.chi2*1./thing.measured_data_vector.size for thing in self.interpretations])
        for i, this_interp in enumerate(self.interpretations):
            if chi2_list[i] >= self.chi2_dof_threshold: # this is a totally arbitrary choice of number for now
                self.systematics_judgment[i] = True
                print("bad fit, updating systematics")


        if self.is_tension == True:
            # determine how much to increase errors in order to solve tension... here?
            # then compare err_increase to max_err_increase
            # if err_increase >= max_err_increase: cosmology_judgment = True
            temp_tm = self.tm
            temp_thresh = self.tm_thresh

            #loop through...
            biasfactor = 0
            while ((np.sum(temp_tm > temp_thresh) > 0) and (biasfactor <= self.max_bias)):
                biasfactor += 0.1
                for i, this_interp in enumerate(self.interpretations[1:]):
                    diff_vec = self.interpretations[0].best_fit_cosmological_parameters - this_interp.best_fit_cosmological_parameters
                    joint_sum_cov = ((1+biasfactor)*self.interpretations[0].best_fit_cosmological_parameter_covariance + (1+biasfactor)*this_interp.best_fit_cosmological_parameter_covariance)
                    # chisq difference in matrix form
                    nsample = 10000
                    samples = np.random.multivariate_normal(mean=np.zeros(this_interp.best_fit_cosmological_parameters.size),cov=joint_sum_cov,size=nsample)
                    logL = np.zeros(nsample)
                    for j in range(nsample):
                        this_diff = samples[j,:]
                        logL[j] = np.matmul(np.matmul(np.transpose(this_diff), np.linalg.inv(joint_sum_cov)), this_diff)
                    temp_tm[i+1] = np.matmul(np.matmul(np.transpose(diff_vec), np.linalg.inv(joint_sum_cov)), diff_vec)
                    temp_thresh = np.percentile(logL,99.9) #/100. # RH changed this by hand
                    print(f"tension metric: {temp_tm} for biasfactor of {biasfactor}")
            print(f"bias factor {biasfactor} either resolves the tension or is too high compared to the max threshold {self.max_bias}!")


            if (biasfactor > self.max_bias):
                print(f"error increase that solves tension is way too much; updating cosmology")
                self.cosmology_judgment = True
            else:
                print(f"No tension if you increase parameter covariance by factor of {biasfactor}. Yay!")
                print("Updating interpreters.best_fit_cosmological_parameter_covariance accordingly")
                for interpreter in self.interpretations:
                    interpreter.best_fit_cosmological_parameter_covariance = interpreter.best_fit_cosmological_parameter_covariance*(1+biasfactor)
                # this means in the next call to tension_metric it will use new covariance and find no tension hopefully
                self.is_tension = False
                self.tm = temp_tm


        self._update_parameters()


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

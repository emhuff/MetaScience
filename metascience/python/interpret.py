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
    def evaluate_logL(parameters):
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
    def _add_systematics():
        pass

    @abstractmethod
    def elaborate_systematics():
        '''
        Fit a model. Generate a posterior.
        '''
        pass

class SimplePendulumExperimentInterpreter(ExperimentInterpreter):
    def __init__(self,
                    times = None,
                    measured_data_vector = None,
                    cosmology_parameters = None,
                    nuisance_parameters = None,
                    experimental_parameters = None,
                    noise_parameters = None,
                    systematics_parameters = None ):

        super().__init__()
        self.measured_data_vector = measured_data_vector
        self.covariance = np.eye(np.ones_like(self.measured_data_vector) + noise_parameters['noise_std_dev']**2)
        self.times = times
        self.best_fit_parameters = None

    def _add_systematics(a_data_vector):
        number_of_functions = len(self.systematics_parameters)
        

        return new_data_vector


    def evaluate_logL(parameters):
        '''
        What can we infer about the world from the provided experiment?
        Takes a cosmology, returns a log-posterior or similar.
        '''
        constant_g = parameters[0]
        constant_l = parameters[1]
        constant_theta_0 = parameters[2]
        model_data_vector = constant_theta_0 * np.cos(np.sqrt(constant_g / constant_l) * self.times)
        model_with_systematics = self._add_systematics(model_data_vector)

        chisq = np.dot(model_with_systematics - self.measured_data_vector,)
        return chisq

    def fit_model():
        '''
        Fit a model. Generate a posterior.
        '''
        self.best_fit_parameters = scipy.optimize.root(evaluate_logL posterior,method='lm')

    def elaborate_systematics():

        # some conditional logic here
        # then:
        def _add_systematics():
            # as a new kind of systematics function
            pass

        pass

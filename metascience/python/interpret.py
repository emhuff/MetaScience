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

    @abstractmethod
    def check_inputs():
        '''
        check that the interpreter has the inputs that are appropriate for then
        experiment
        check that the data vector from the experiment module match interpret
        assert?
        '''
        pass

class SimplePendulumExperimentInterpreter(ExperimentInterpreter):
    def __init__(self,
                 experiment = None,
                 cosmology_parameters = None,
                 nuisance_parameters = None,
                 noise_parameters = None,
                 systematics_parameters = None ):

        '''
        notes:
            why doesn't covariance include all errors (i.e., systematics)?
            I don't think we need cosmo, nuisance, expt params as inputs

            ... times: input data
            ...
        '''

        super().__init__()
        self.kind = 'pendulum'
        self.measured_data_vector = measured_data_vector
        self.covariance = np.eye(np.ones_like(self.measured_data_vector)
            + noise_parameters['noise_std_dev']**2)
        self.times = times
        self.best_fit_parameters = None


    def _add_systematics(a_data_vector):
        '''
        Adding terms in the Hankel basis functions to approximate systematics
        Adding systematics here that don't match the input systematics, becuase
        ... we don't know what they are!
        '''
        if self.systematics_parameters['function'] == 'hankel':
            order_of_function = len(self.systematics_parameters['coeff'])
            added_systematics_vector = scipy.special.hankel1(order_of_function, a_data_vector)
            new_data_vector = a_data_vector + added_systematics_vector
        else:
            raise NotImplementedError

        return new_data_vector

    def evaluate_logL(parameters):
        '''
        What can we infer about the world from the provided experiment?
        Takes a cosmology, returns a log-posterior or similar.
        notes:
            is the syntax right for something dotted with itself?
            is the covariance attached in the right way?
        '''
        constant_g = parameters[0]
        constant_l = parameters[1]
        constant_theta_0 = parameters[2]
        model_data_vector = constant_theta_0 * np.cos(np.sqrt(constant_g /
            constant_l) * self.times)
        model_with_systematics = self._add_systematics(model_data_vector)
        delta = model_with_systematics - self.measured_data_vector
        chisq = np.dot(delta,delta) / self.covariance

        return chisq

    def fit_model(guess):
        '''
        Fit a model to generate a posterior of best-fit parameters
        notes:
            maybe we need an explicit jacobian because the model has cosine in it?
        '''
        self.best_fit_parameters = scipy.optimize.root(evaluate_logL, guess,
            method='lm')
        return self.best_fit_parameters


    def elaborate_systematics():

        # some conditional logic here
        # then:
        def _add_systematics():
            # as a new kind of systematics function
            pass

        pass

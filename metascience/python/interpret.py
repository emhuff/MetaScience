from abc import ABCMeta, abstractmethod
import numpy as np

class ExperimentInterpreter(metaclass=ABCMeta):
    def __init__(self):
        self.kind = None
        pass

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
    def _check_inputs():
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
                 systematics_parameters = None,
                 noise_parameters = None,
                 prior = None
                 ):

        '''
        notes:
            why doesn't covariance include all errors (i.e., systematics)?
            I don't think we need cosmo, nuisance, expt params as inputs

            ... times: input data
            ...
            inputs:
                parameters are all dictionaries
                experiment: object
                prior: object


        '''

        super().__init__()
        self.kind = 'pendulum'
        self.measured_data_vector = experiment.data_vector
        self.covariance = np.diag(np.zeros_like(self.measured_data_vector)
            + noise_parameters['noise_std_dev']**2)
        self.times = experiment.times
        ## The below parameters are stored by name in the experiment module.
        self.best_fit_cosmological_parameters = cosmology_parameters
        self.best_fit_nuisance_parameters = nuisance_parameters
        self.fit_status = None

        self._cosmology_parameter_names = self.cosmology_parameters.keys()
        self._nuisance_parameter_names = self.nuisance_parameters.keys()
        self._systematics_parameter_names = self.systematics_parameters.keys()
        self._parameter_names = [self._cosmology_parameter_names,
            self._nuisance_parameter_names,
            self._systematics_parameter_names]

    def _generate_guess(self):
        c_guess = [self.cosmology_parameters[key] for key in self._cosmology_parameter_names]
        n_guess = [self.nuisance_parameters[key] for key in self._nuisance_parameter_names]
        s_guess = [self.systematics_parameters[key] for key in self._systematics_parameter_names]
        guess = np.concatenate([c_guess,n_guess,s_guess])

    def _check_inputs(experiment, cosmology_parameters, nuisance_parameters,
        noise_parameters, systematics_parameters):
        '''
        check specific input for the interpretation of this experimental
        ... existence of particular parameters in dictionaries
        ... types of data for these parameters
        ... dimensions of data data for these parameters
        ... experiment is instance of the relevant class?
        '''

        # check experiment isinstance
        #assert(isinstance(experiment, SimplePendulumExperiment))

        #assert(cosmology_)


    def _add_systematics(a_data_vector):
        '''
        Adding terms in the Hankel basis functions to approximate systematics
        Adding systematics here that don't match the input systematics, becuase
        ... we don't know what they are!
        '''
        if self.systematics_parameters['function'] == 'hankel':
            order_of_function = len(self.systematics_parameters['coeff'])
            added_systematics_vector = scipy.special.hankel1(order_of_function,
                a_data_vector)
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



        # identify parameters by name to connect with functional form in th emodel
        constant_g = parameters[ self._parameter_names.index('constant_g') ]
        constant_l = parameters[ self._parameter_names.index('constant_l') ]
        constant_theta_0 = parameters[ self._parameter_names.index('constant_theta_0') ]
        constant_phase = parameters[ self._parameter_names.index('constant_phase') ]

        # define model for data with parameters above
        model_data_vector = constant_theta_0 * np.cos(np.sqrt(constant_g /
            constant_l) * self.times + constant_phase)

        # add systematics to the model
        model_with_systematics = self._add_systematics(model_data_vector)

        # calculate difference between model and data
        delta = model_with_systematics - self.measured_data_vector

        # calcualte chisq
        chisq = np.dot(delta,delta) / self.covariance

        return chisq

    def fit_model(self):
        '''
        Fit a model to generate a posterior of best-fit parameters
        notes:
            maybe we need an explicit jacobian because the model has cosine in it?
        '''
        # generate a guess in the right order.
        guess = self._generate_guess()

        # fit for parameters
        best_fit_parameters = scipy.optimize.root(evaluate_logL, guess,
            method = 'lm')

        # generate list of paraemters for each kind of parameter in teh correct order
        for i,name in enumerate(self._cosmology_parameter_names):
            self.best_fit_cosmological_parameters[name] = best_fit_parameters.x[self._parameter_names.index(name)]
        for i,name in enumerate(self._nuisance_parameter_names):
            self.best_fit_nuisance_parameters[name] = best_fit_parameters.x[self._parameter_names.index(name)]
        for i,name in enumerate(self._systematics_parameter_names):
            self.best_fit_systematics_parameters[name] = best_fit_parameters.x[self._parameter_names.index(name)]

        # apply success flag from fit parameters to fit status
        self.fit_status = best_fit_parameters.success

    def elaborate_systematics(self):

        # some conditional logic here
        # then:
        def _add_systematics():
            # as a new kind of systematics function
            pass

        pass

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy

#import cosmology ???

class ExperimentInterpreter(metaclass=ABCMeta):
    def __init__(self):
        self.kind = None
        pass

    '''
    @abstractmethod
    def evaluate_logL(parameters):
        #
        #What can we infer about the world from the provided experiment?
        #Takes a cosmology, returns a log-posterior or similar.
        #
        #raise NotImplementedError
    '''

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


class CargoCultExperimentInterpreter(ExperimentInterpreter):
    def __init__(self,
                 experiment = None,
                 parameters = None,
                 systematics_parameters = None,
                 noise_parameters = None,
                 prior = None,
                 cosmology = None
                 ):

        super().__init__()

        self.best_fit_cosmological_parameters = parameters
        self.best_fit_cosmological_parameter_covariance = np.eye(cosmology.n_parameters)
        self.chi2 = 0.
        self.times = np.arange(10)

    def fit_model():
        '''
        Fit a model. Generate a posterior.
        '''
        parameters = cosmology.get_parameter_set()

        model_data_vector = cosmology.generate_model_data_vector(self.times,parameters[:cosmology.n_parameters])
        # add systematics to the model
        model_with_systematics = self._add_systematics(model_data_vector,pars = parameters[cosmology.n_parameters:])

        best_fit_parameters = self.best_fit_cosmological_parameters# cosmology.best_fit_parameters

        return best_fit_parameters

    def _add_systematics():
        pass

    def elaborate_systematics():
        '''
        Fit a model. Generate a posterior.
        '''
        pass


    def _check_inputs():
        '''
        check that the interpreter has the inputs that are appropriate for the
        experiment
        check that the data vector from the experiment module match interpret
        assert?
        '''
        pass


class SimplePendulumExperimentInterpreter(ExperimentInterpreter):
    def __init__(self,
                 experiment = None,
                 starting_systematics_parameters = None,
                 noise_parameters = None,
                 prior = None,
                 cosmology = None,
                 ):

        '''
        notes:
            why doesn't covariance include all errors (i.e., systematics)?
            I don't think we need cosmo, nuisance, expt params as inputs

            ... times: input data
            ...
            inputs:
                parameters is a list of Parameter objects.
                experiment: object
                prior: object
        '''

        super().__init__()
        self.kind = 'pendulum'
        self.measured_data_vector = experiment.observed_data_vector
        self.covariance = np.diag(np.zeros_like(self.measured_data_vector)
            + noise_parameters[0]**2)
        self.inverse_covariance = np.linalg.inv(self.covariance)
        self.times = experiment.times
        ## The below parameters are stored by name in the experiment module.
        self.starting_systematics_parameters = starting_systematics_parameters
        self.cosmology = cosmology
        self.best_fit_cosmological_parameters = None
        self.best_fit_nuisance_parameters = None
        self.best_fit_systematics_parameters = None

        self.fit_status = None


    def _add_systematics(self,data_vector, parameters = None):
        '''
        Adding terms in the Hankel basis functions to approximate systematics
        Adding systematics here that don't match the input systematics, becuase
        ... we don't know what they are!
        '''
        '''
        if self.best_fit_systematics_parameters['function'] == 'hankel':
            order_of_function = len(self.best_fit_systematics_parameters['coeff'])
            added_systematics_vector = scipy.special.hankel1(order_of_function,
                a_data_vector)
            new_data_vector = a_data_vector + added_systematics_vector
        else:
            raise NotImplementedError
        '''
        order_of_function = len(parameters)
        added_systematics_vector = scipy.special.hankel1(order_of_function,data_vector)
        return data_vector + added_systematics_vector

    def fit_model(self):
        '''
        Fit a model to generate a posterior of best-fit parameters
        notes:
            maybe we need an explicit jacobian because the model has cosine in it?
        '''

        def evaluate_logL(parameters):
            '''
            What can we infer about the world from the provided experiment?
            Takes a cosmology, returns a log-posterior or similar.
            notes:
                is the syntax right for something dotted with itself?
                is the covariance attached in the right way?
            '''
            model_data_vector = self.cosmology.generate_model_data_vector(self.times,parameters[:self.cosmology.n_parameters])
            # add systematics to the model
            model_with_systematics = self._add_systematics(model_data_vector,parameters = parameters[self.cosmology.n_parameters:])

            # calculate difference between model and data
            delta = model_with_systematics - self.measured_data_vector

            # calcualte chisq

            chisq = np.dot(delta.T, np.dot(self.inverse_covariance,delta))

            return chisq


        # generate a guess in the right order.
        guess = np.concatenate([self.cosmology.best_fit_cosmological_parameters,self.cosmology.best_fit_nuisance_parameters,self.starting_systematics_parameters])

        # fit for parameters
        best_fit_parameters = scipy.optimize.root(evaluate_logL, guess, method = 'lm')

        chi2 = fit_model.evaluate_logL(best_fit_parameters)/len(self.measured_data_vector) # save the best-fit chi2

        # generate list of paraemters for each kind of parameter in the correct order
        # ..todo: make this consistent with new parameters definition.
        self.best_fit_cosmological_parameters = best_fit_parameters.x[:self.cosmology.n_cosmological]
        self.best_fit_nuisance_parameters = best_fit_parameters.x[self.cosmology.n_cosmological:self.cosmology.n_parameters]
        self.best_fit_systematics_parameters = best_fit_parameters.x[self.cosmology.n_parameters:]

        # apply success flag from fit parameters to fit status
        self.fit_status = best_fit_parameters.success





    def elaborate_systematics(self):

        # some conditional logic here
        # then:
        def _add_systematics():
            # as a new kind of systematics function
            pass

        pass

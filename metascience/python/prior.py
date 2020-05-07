from abc import ABCMeta, abstractmethod
import numpy as np

class prior(metaclass=ABCMeta):
    def _init__(self):
        self.kind = 'base'
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def draw(self):
        pass

class gaussian_prior(prior):
    def __init__(self, parameter_means = None, parameter_names, covariance = None):
        '''
        A class to hold, evaluate, and draw from a gaussian prior.
        inputs:
            parameter_means: mean values 
            covariance:

        '''
        self.means = means
        self.covariance = covariances
        try:
            self.inverse_covariance = np.linalg.inv(covariance)
        except:
            raise Exception("provided covariance should be invertable")
        pass

    def evaluate_log(self, values):
        '''
        Evaluate a log of normal distribution
        inputs:
            values
        outputs:
            prior (in log)
        '''
        sgn_determ, log_determ = np.slogdet(self.covariance)
        log_prior = -np.dot((values - self.means).T,
            self.inverse_covariance).dot(values-self.means) / 2. -
            sgn_determ* len(self.mean) / 2. * log_determ
        return log_prior

    def evaluate(self, values):
        '''
        Evaluate a normal distribution
        inputs:
            values: a dictionary of parameter-name/value pairs
        outputs:
            prior
        '''
        sgn_determ, log_determ = np.slogdet(self.covariance)
        log_prior = -np.dot((values - self.means).T,
            self.inverse_covariance).dot(values - self.means) / 2. -
            sgn_determ * len(self.mean) / 2. * log_determ
        return np.exp(log_prior)

    def draw(self, number_of_draws = 1):
        '''
        Draw from a multivariate multivariate
        inputs:
            number of draws
        outputs:
            values drawn from the distribution
        '''
        draws = np.random.multivariate_normal(self.means,
            self.covariance, size=number_of_draws)
        return draws

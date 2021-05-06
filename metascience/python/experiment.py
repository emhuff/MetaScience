from abc import ABCMeta, abstractmethod,abstractproperty
import numpy as np
import scipy
from scipy.integrate import solve_ivp as solver
# For debugging:
import ipdb
import matplotlib.pyplot as plt

class Experiment(metaclass=ABCMeta):
    def __init__(self):
        self.kind = None
        self.ideal_data_vector = None
        self.observed_data_vector = None
        pass

    @abstractmethod
    def _get_ideal_data_vector(self):
        # generate the ideal data vector from the cosmology.
        pass

    @abstractmethod
    def _add_systematics(self):
        pass

    @abstractmethod
    def _add_noise(self):
        pass

    def generate_data(self):
        '''
        Generate data based on a few functions designed in the class
        '''
        self._get_ideal_data_vector()
        self._add_noise()
        self._add_systematics()


class CMBExperiment(Experiment):
    '''
    This class instantiates an experiment where we:
        - have measurements of CMB Tempterature taken at points in the sky.
    '''
    def __init__(self,cosmology = None, systematics_model = None, noise_model = None, experimental_parameters = None):
        super().__init__()
        self.kind = 'CMB'
        self.cosmology = cosmology

    def _set_systematics(systematics_model):
        pass

    def _set_noise(noise_model):
        pass


class SimplePendulumExperiment(Experiment):
    '''
    This class instantiates an experiment where we:
      - have measurements of the position of pendulum, theta, taken at certain exact times

    '''
    def __init__(self,
                    cosmology_parameters = None,
                    nuisance_parameters = None,
                    experimental_parameters = None,
                    cosmology = None,
                    noise_parameters = None,
                    systematics_parameters = None,
                    lowest_systematics_coeff = 500,
                    seed=999):
        super().__init__()
        self.kind = 'pendulum'
        self.parameters = np.concatenate([cosmology_parameters,nuisance_parameters])
        # check that this is consistent with what the cosmology needs.
        assert cosmology.n_cosmological == cosmology_parameters.size
        assert cosmology.n_nuisance == nuisance_parameters.size
        self.cosmology = cosmology
        self.times = experimental_parameters['times']
        self.systematics_parameters = systematics_parameters
        self.noise_parameters = noise_parameters
        self.lowest_systematics_coeff = lowest_systematics_coeff


        self.seed = np.random.seed(seed=seed)

        #"cosmology" params: g
        #"astrophysical (nuisance)"  params: l

    def _get_ideal_data_vector(self):
        '''
        Generate ideal data vector from cosmology parameters, nuisance
        parameters, and experimental parameters
        '''
        self.ideal_data_vector = self.cosmology.generate_model_data_vector(self.times,self.parameters)
        self.observed_data_vector = self.ideal_data_vector

    def _add_noise(self):
        '''
        Add noise from noise parrrrrameters and measurements
        '''
        noise_std_dev = self.noise_parameters[0]
        noise_vector = noise_std_dev * np.random.randn(self.times.size)
        self.observed_data_vector = self.observed_data_vector + noise_vector

    def _add_systematics(self):
        '''
        This lets the experiment generate systematics that are guaranteed
        to be (eventually) successfully modeled by the interpreter, if it is
        also using Hankel functions to model the systematics.
        This function picks coefficients of the Hankel expansion at random,
        with amplitudes that in general decrease as they go to higher order.
        Fitting the first N terms is thus always a good but limited
        approximation to the true model.

        Adding terms in the Hankel basis functions to approximate systematics
        '''
        # Check to see if systematics parameters exist.
        parameters = self.systematics_parameters
        added_systematics_vector = np.zeros_like(self.times)
        self.systematics_vector = np.zeros_like(self.observed_data_vector)
        for nu,coeff in enumerate(parameters):
            #thissys = coeff*scipy.special.hankel1(nu,self.times/np.max(self.times)*2*np.pi)
            arg = self.times / (np.max(self.times) - np.min(self.times))
            arg = arg - np.min(arg)
            thissys = coeff*scipy.special.eval_laguerre(2*nu+1+self.lowest_systematics_coeff,arg)
            if np.sum(~np.isfinite(thissys)) > 0:
                if nu == 0:
                    thissys[~np.isfinite(thissys)] = coeff
                else:
                    thissys[~np.isfinite(thissys)] = 0.
            self.systematics_vector = self.systematics_vector + thissys.real
        self.observed_data_vector = self.observed_data_vector + self.systematics_vector

    def _add_systematics_devilish(self):
        '''
        Creating a 'boost' by multiplying the normal linear gradient of data/time
        by a boost (driving) factor
        adding a systematic error
        '''
        boost_deriv = self.systematics_parameters[0]
        time_deriv = self.times - self.times[::-1] # compute the time deriv vector
        data_deriv = self.ideal_data_vector - self.ideal_data_vector[::-1] #compute the data deriv vector
        systematics_vector = boost_deriv * (data_deriv / time_deriv) # multiply the ratio of ddata/dtime by boost factor
        self.observed_data_vector = self.observed_data_vector + systematics_vector




class DistanceModulusExperiment(Experiment):
    '''
    This class instantiates an experiment where we:
      - have measurements of the redshift, imported as 'times' for the code

    '''
    def __init__(self,
                    cosmology_parameters = None,
                    nuisance_parameters = None,
                    experimental_parameters = None,
                    cosmology = None,
                    noise_parameters = None,
                    systematics_parameters = None,
                    lowest_systematics_coeff = 500,
                    seed=999):
        super().__init__()
        self.kind = 'distmod'
        self.parameters = np.concatenate([cosmology_parameters,nuisance_parameters])
        # check that this is consistent with what the cosmology needs.
        assert cosmology.n_cosmological == cosmology_parameters.size
        assert cosmology.n_nuisance == nuisance_parameters.size
        self.cosmology = cosmology
        self.times = experimental_parameters['redshifts']
        self.systematics_parameters = systematics_parameters
        self.noise_parameters = noise_parameters
        self.lowest_systematics_coeff = lowest_systematics_coeff


        self.seed = np.random.seed(seed=seed)

        #"cosmology" params: g
        #"astrophysical (nuisance)"  params: l

    def _get_ideal_data_vector(self):
        '''
        Generate ideal data vector from cosmology parameters, nuisance
        parameters, and experimental parameters
        '''
        self.ideal_data_vector = self.cosmology.generate_model_data_vector(self.times,self.parameters)
        self.observed_data_vector = self.ideal_data_vector

    def _add_noise(self):
        '''
        Add noise from noise parrrrrameters and measurements
        '''
        noise_std_dev = self.noise_parameters[0]
        noise_vector = noise_std_dev * np.random.randn(self.times.size)
        self.observed_data_vector = self.observed_data_vector + noise_vector

    def _add_systematics(self):
        '''
        This lets the experiment generate systematics that are guaranteed
        to be (eventually) successfully modeled by the interpreter, if it is
        also using Hankel functions to model the systematics.
        This function picks coefficients of the Hankel expansion at random,
        with amplitudes that in general decrease as they go to higher order.
        Fitting the first N terms is thus always a good but limited
        approximation to the true model.

        Adding terms in the Hankel basis functions to approximate systematics
        '''
        # Check to see if systematics parameters exist.
        parameters = self.systematics_parameters
        added_systematics_vector = np.zeros_like(self.times)
        self.systematics_vector = np.zeros_like(self.observed_data_vector)
        for nu,coeff in enumerate(parameters):
            #thissys = coeff*scipy.special.hankel1(nu,self.times/np.max(self.times)*2*np.pi)
            arg = self.times / (np.max(self.times) - np.min(self.times))
            arg = arg - np.min(arg)
            thissys = coeff*scipy.special.eval_laguerre(2*nu+1+self.lowest_systematics_coeff,arg)
            if np.sum(~np.isfinite(thissys)) > 0:
                if nu == 0:
                    thissys[~np.isfinite(thissys)] = coeff
                else:
                    thissys[~np.isfinite(thissys)] = 0.
            self.systematics_vector = self.systematics_vector + thissys.real
        self.observed_data_vector = self.observed_data_vector + self.systematics_vector

    def _add_systematics_devilish(self):
        '''
        Creating a 'boost' by multiplying the normal linear gradient of data/time
        by a boost (driving) factor
        adding a systematic error
        '''
        boost_deriv = self.systematics_parameters[0]
        time_deriv = self.times - self.times[::-1] # compute the time deriv vector
        data_deriv = self.ideal_data_vector - self.ideal_data_vector[::-1] #compute the data deriv vector
        systematics_vector = boost_deriv * (data_deriv / time_deriv) # multiply the ratio of ddata/dtime by boost factor
        self.observed_data_vector = self.observed_data_vector + systematics_vector

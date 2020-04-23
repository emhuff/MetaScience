from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp as solver

class Experiment():
    def __init__(self):
        pass

    @property
    def cosmology():
        '''
        Need to specify the cosmology.
        '''
        raise NotImplementedError()

    @property
    def systematics():
        '''
        Need to specify the systematic error model.
        '''
        raise NotImplementedError()

    @property
    def noise():
        '''
        Need to specify the noise model.
        '''
        raise NotImplementedError()

    @property
    def kind():
        '''
        What experiment is this? Know thyself, object.
        '''

    @abstractmethod
    def _get_ideal_data_vector():
        # generate the ideal data vector from the cosmology.
        pass

    @abstractmethod
    def _add_systematics():
        pass

    @abstractmethod
    def _add_noise():
        pass

    def generate_data():
        '''
        Generate data based on a few functions designed in the class
        '''
        self._get_ideal_data_vector()
        self._add_noise()
        self.data_vector = self._add_systematics()


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
                    noise_parameters = None,
                    systematics_parameters = None,
                    seed=999):
        super().__init__()
        self.kind = 'pendulum'
        self.constant_g = cosmology_parameters['constant_g']
        self.constant_l = nuisance_parameters['constant_l']
        self.constant_theta_0 = nuisance_parameters['constant_theta_0']
        self.constant_theta_v0 = nuisance_parameters['constant_theta_v0']
        self.time_between_measurements = experimental_parameters['time_between_measurements']
        self.number_of_measurements = experimental_parameters['number_of_measurements']
        self.times = None
        self.noise_std_dev = noise_parameters['noise_std_dev']
        self.boost_deriv = systematics_parameters['boost_deriv'] # what's a boost deriv?
        self.seed = np.random.seed(seed=seed)
        self.systematics_parameters = systematics_parameters
        #"cosmology" params: g
        #"astrophysical (nuisance)"  params: l

    def _get_ideal_data_vector():
        '''
        Generate ideal data vector from cosmology parameters, nuisance parameters, and experimental parameters
        '''
        g = self.constant_g
        l = self.constant_l

        c = self.systematics_parameters['drag_coeff']
        A = self.systematics_parameters['driving_amp']
        wd = self.systematics_parameters['driving_freq']
        phid = self.systematics_parameters['driving_phase']

        self.times = np.arange(self.number_of_measurements) * self.time_between_measurements
        #theta = self.constant_theta_0 *
        #    np.cos(np.sqrt(self.constant_g] / self.constant_l) * self.times)

        def forcing_function(t):
            return A*np.cos(wd*t+phid)

        def oscillator_eqns(t,y):
            x = y[0]
            u = y[1]
            xp = u
            up = (forcing_function(t) - c*u - l*x)/g
            return np.array([xp,up])

        interval = self.number_of_measurements * self.time_between_measurements
        solution = solver(oscillator_eqns,interval, np.array([self.theta_0,self.theta_v0]), t_eval = self.times)

        return solution.y

    def _add_systematics():
        '''
        Creating a 'boost' by multiplying the normal linear gradient of data/time
        by a boost (driving) factor
        adding a systematic error

        '''
        time_deriv = self.times - self.times[::-1] # compute the time deriv vector
        data_deriv = self.data_vector - self.data_vector[::-1] #compute the data deriv vector
        systematics_vector = self.boost_deriv * (data_deriv / time_deriv) # multiply the ratio of ddata/dtime by boost factor
        return self.data_vector + systematics_vector # add systematics to data vector

    def _add_noise():
        '''
        Add noise from noise parrameters and measurements
        '''
        noise_vector = self.noise_std_dev * np.random.randn(self.number_of_measurements)
        self.data_vector = self.ideal_data_vector + noise_vector

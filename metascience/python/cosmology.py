from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp as solver

class Cosmology(metaclass = ABCMeta):
    def __init__(self,complexity):
        self.n_parameters = False
        self.n_cosmological = False
        self.n_nuisance = False
        self.name = 'base cosmology. Probably you should actually give me a name.'
        pass

    @abstractmethod
    def get_parameter_set(self):
        pass

    @abstractmethod
    def generate_model_data_vector(self,):
        pass



class CargoCultCosmology(Cosmology):
    def __init__(self):
        self.complexity = 0
        self.n_cosmological = 2
        self.n_nuisance = 0
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.zeros(self.n_cosmological)
        self.fiducial_nuisance_parameters = np.zeros(self.n_cosmological)


    def get_parameter_set(self):
         parameters = np.zeros(self.n_cosmological)
         return parameters


    def generate_model_data_vector(self,times,parameters):
#        parameters = get_parameter_set()
        model_data_vector = parameters[0]*np.ones(len(times))**2 + parameters[1]

        return model_data_vector

class CargoCultCosmology_Ones(Cosmology):
    def __init__(self):
        self.complexity = 0
        self.n_cosmological = 2
        self.n_nuisance = 0
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.zeros(self.n_cosmological)+1.0
        self.fiducial_nuisance_parameters = np.zeros(self.n_nuisance)
#        self.best_fit_cosmological_parameters = np.zeros(self.n_parameters)
#        self.best_fit_cosmological_parameter_covariance=np.eye(n_parameters)
#        self.chi2 = 0.

    def get_parameter_set(self):
         parameters = np.zeros(self.n_cosmological)+1.0

         return parameters

    def generate_model_data_vector(self,times,parameters):
#        self.best_fit_cosmological_parameters = get_parameter_set()
        model_data_vector = parameters[0]*np.ones(len(times)) + parameters[1]

        return model_data_vector


class CargoCultCosmology_Tens(Cosmology):
    def __init__(self):
        self.complexity = 0
        self.n_cosmological = 2
        self.n_nuisance = 0
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.zeros(self.n_cosmological)+10.0
        self.fiducial_nuisance_parameters = np.zeros(self.n_nuisance)

    def get_parameter_set(self):
         parameters = np.zeros(self.n_cosmological)+10.0

         return parameters

    def generate_model_data_vector(self,times,parameters):
#        self.best_fit_cosmological_parameters = get_parameter_set()
        model_data_vector = parameters[0]*np.ones(len(times)) + parameters[1]**2

        return model_data_vector

class StraightLineCosmology(Cosmology):
    def __init__(self):
        self.complexity = 0
        self.n_cosmological = 1
        self.n_nuisance = 1
        self.name = 'straight-line cosmology'
        self.n_parameters = self.n_cosmological + self.n_nuisance
        self.fiducial_cosmological_parameters = np.array([1.])
        self.fiducial_nuisance_parameters = np.array([0.])

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

    def generate_model_data_vector(self,times,parameters):
        slope = parameters[0]
        intercept = parameters[1]

        # for now assuming a straight line with the intercept the nuisance parameter that isn't the cosmology
        model_data_vector =slope*times + intercept
        return model_data_vector



class ExponentialCosmology(Cosmology):
    def __init__(self):
        self.complexity = 0
        self.n_cosmological = 1
        self.n_nuisance = 1
        self.name = 'exponential cosmology'
        self.n_parameters = self.n_cosmological + self.n_nuisance
        self.fiducial_cosmological_parameters = np.array([1.])
        self.fiducial_nuisance_parameters = np.array([0.])

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

    def generate_model_data_vector(self,times,parameters):
        timescale = parameters[0]
        amplitude = parameters[1]

        # for now assuming a straight line with the intercept the nuisance parameter that isn't the cosmology
        model_data_vector = amplitude*np.exp(-np.abs(times)/np.abs(timescale+1.))
        return model_data_vector


class CosineCosmology(Cosmology):
    def __init__(self):
        self.complexity = 1
        self.n_cosmological = 1
        self.n_nuisance = 2
        self.name = "Cosine cosmology"
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.array([.50]) # frequency
        self.fiducial_nuisance_parameters = np.array([2.0,.0]) # amplitude, phase

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

    def generate_model_data_vector(self,times, parameters = None):
        # define model for data with parameters above

        constant_w = parameters[0]
        constant_theta_0 = parameters[1]
        constant_phase = parameters[2]

        model_data_vector = constant_theta_0*np.cos(constant_w * times + constant_phase)
        return model_data_vector



class TrueCosmology(Cosmology):
    def __init__(self):
        self.n_cosmological = 1
        self.n_nuisance = 6
        self.name = 'Damped-driven harmonic oscillator cosmology'
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.array([1.0]) # w
        self.fiducial_nuisance_parameters = np.array([0.10,0.20,0.3,np.pi,.0,1.0])
#        self.fiducial_nuisance_parameters = np.array([0.0,0.0,0.3,np.pi,.0,1.0]) # to turn off damping and driving

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

# copied from experiment.py
    def generate_model_data_vector(self, times, parameters = None):
        '''
        Generate ideal data vector from cosmology parameters, nuisance
        parameters, and experimental parameters
        '''
        w = parameters[0]
        c = parameters[1]
        A = parameters[2] # amplitude of driver
        wd = parameters[3] # frequency of driver
        phid = parameters[4] # phase of driver
        theta_x0 = parameters[5]
        theta_v0 = parameters[6]

        #theta = self.constant_theta_0 *
        #    np.cos(np.sqrt(self.constant_g] / self.constant_l) * self.times)

        def forcing_function(t):
            '''
            output amplitude as a function of time
            '''
            return A*np.cos(wd*t+phid)

        def oscillator_eqns(t,y):
            '''
            general equations of motion for an oscillator
            '''
            x = y[0]
            u = y[1]
            xp = u
            up = (forcing_function(t) - c * u - w**2 * x)
            return np.array([xp,up])

        # minimum and maximum times over which the solution should be calculated
        interval = (np.min(times),np.max(times))

        # solving the oscillator equations of motion for the total interval
        solution = solver(oscillator_eqns, interval, np.array([parameters[-2],parameters[-1]]), t_eval = times)
        self.ideal_data_vector = solution.y[0]
        return solution.y[0]

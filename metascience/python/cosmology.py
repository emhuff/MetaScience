from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp as solver
from scipy import special as sp
from astropy.cosmology import LambdaCDM

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


class LCDM_distanceModulus(Cosmology):
    def __init__(self):
        self.complexity = 0
        self.n_cosmological = 4
        self.n_nuisance = 0
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.zeros(self.n_cosmological)
        self.fiducial_nuisance_parameters = np.zeros(self.n_cosmological)

    def get_parameter_set(self):
         parameters = np.zeros(self.n_cosmological)
         return parameters

    def generate_model_data_vector(self,times,parameters):
#        parameters = get_parameter_set()

        cosmo = LambdaCDM(H0=parameters[0], Om0=parameters[1], Ode0=parameters[2], Ob0=parameters[3], Tcmb0=2.725)
        model_data_vector = cosmo.distmod(times).value
        #not keeping units here

        return model_data_vector

class LCDM_age(Cosmology):
    def __init__(self):
        self.complexity = 0
        self.n_cosmological = 4
        self.n_nuisance = 0
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.zeros(self.n_cosmological)
        self.fiducial_nuisance_parameters = np.zeros(self.n_cosmological)

    def get_parameter_set(self):
         parameters = np.zeros(self.n_cosmological)
         return parameters

    def generate_model_data_vector(self,times,parameters):
#        parameters = get_parameter_set()

        cosmo = LambdaCDM(H0=parameters[0], Om0=parameters[1], Ode=parameters[2], Ob0=parameters[3], Tcmb0=2.725)
        model_data_vector = cosmo.age(times).value
        #not keeping units here

        return model_data_vector


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
        self.cosmological_parameter_names = ['inverse slope']
        self.fiducial_nuisance_parameters = np.array([0.])
        self.nuisance_parameter_names = ['intercept']

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

    def generate_model_data_vector(self,times,parameters):
        slope = 1./parameters[0]
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
        self.cosmological_parameter_names = ['inverse decay time of exponential']
        self.fiducial_nuisance_parameters = np.array([1.])
        self.nuisance_parameter_names = ['amplitude']

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

    def generate_model_data_vector(self,times,parameters):
        timescale = 1./parameters[0]
        amplitude = parameters[1]

        # for now assuming a straight line with the intercept the nuisance parameter that isn't the cosmology
        model_data_vector = amplitude*np.exp(-np.abs(times)/np.abs(timescale+1.))
        return model_data_vector

class AiryCosmology(Cosmology):
    def __init__(self):
        self.complexity = 1
        self.n_cosmological = 1
        self.n_nuisance = 2
        self.name = "Airy function cosmology"
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.array([.50]) # frequency
        self.cosmological_parameter_names = ['frequency of Airy function']
        self.fiducial_nuisance_parameters = np.array([1.0,.0]) # amplitude, phase
        self.nuisance_parameter_names = ['amplitude','phase']

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

    def generate_model_data_vector(self,times, parameters = None):
        # define model for data with parameters above

        constant_w = parameters[0]
        constant_theta_0 = parameters[1]
        constant_phase = parameters[2]

        model_data_vector = constant_theta_0*sp.airy(-constant_w * times + constant_phase)[0]
        return model_data_vector

class GaussianCosmology(Cosmology):
    def __init__(self):
        self.complexity = 1
        self.n_cosmological = 1
        self.n_nuisance = 4
        self.name = "Gaussian function"
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.array([.50])
        self.cosmological_parameter_names = ['inverse width of gaussian']
        self.fiducial_nuisance_parameters = np.array([1.0,1.0, 1.0])
        self.nuisance_parameter_names = ['constant offset','amplitude','phase']

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

    def generate_model_data_vector(self,times, parameters = None):
        # define model for data with parameters above

        constant_w = parameters[0]
        constant_theta_0 = parameters[1]
        constant_gauss_amp = parameters[2]
        constant_phase = parameters[3]

        model_data_vector = constant_theta_0 - constant_gauss_amp*np.exp(-constant_w**2*(times-constant_phase)**2/2.)/np.sqrt(2*np.pi/constant_w**2)
        return model_data_vector

class BesselJCosmology(Cosmology):
    def __init__(self):
        self.complexity = 1
        self.n_cosmological = 1
        self.n_nuisance = 2
        self.name = "BesselJ cosmology"
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.array([.50]) # frequency
        self.cosmological_parameter_names = ['frequency of oscillator']
        self.fiducial_nuisance_parameters = np.array([1.0,.0]) # amplitude, phase
        self.nuisance_parameter_names = ['amplitude of oscillator', 'phase of oscillator']

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

    def generate_model_data_vector(self,times, parameters = None):
        # define model for data with parameters above

        constant_w = parameters[0]
        constant_theta_0 = parameters[1]
        constant_phase = parameters[2]

        model_data_vector = constant_theta_0*sp.jv(0,constant_w * times + constant_phase)
        return model_data_vector



class CosineCosmology(Cosmology):
    def __init__(self):
        self.complexity = 1
        self.n_cosmological = 1
        self.n_nuisance = 2
        self.name = "Cosine cosmology"
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.array([.50]) # frequency
        self.cosmological_parameter_names = ['frequency of oscillator']
        self.fiducial_nuisance_parameters = np.array([2.0,.0]) # amplitude, phase
        self.nuisance_parameter_names = ['Amplitude of oscillator', 'phase of oscillator']

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



class DampedDrivenOscillatorCosmology(Cosmology):
    def __init__(self):
        self.complexity = 2
        self.n_cosmological = 1
        self.n_nuisance = 6
        self.name = 'Damped-driven harmonic oscillator cosmology'
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.array([1.0]) # w
        self.cosmological_parameter_names = ['oscillator frequency']
        self.fiducial_nuisance_parameters = np.array([.50,0.20,0.3,np.pi,.0,1.0])
#        self.fiducial_nuisance_parameters = np.array([0.0,0.0,0.3,np.pi,.0,1.0]) # to turn off damping and driving
        self.nuisance_parameter_names = ['drag','driver amplitude','driver frequency','driver phase','oscillator initial position','oscillator initial velocity']

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

# copied from experiment.py
    def generate_model_data_vector(self, times, parameters = None):
        '''
        Generate ideal data vector from cosmology parameters, nuisance
        parameters, and experimental parameters
        '''
        w = parameters[0] # frequency of oscillator
        c = parameters[1] # drag parameter
        A = parameters[2] # amplitude of driver
        wd = parameters[3] # frequency of driver
        phid = parameters[4] # phase of driver
        theta_x0 = parameters[5] # initial position of oscillator
        theta_v0 = parameters[6] # initial velocity of oscillator

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

class DampedDrivenOscillatorVariableGCosmology(Cosmology):
    def __init__(self):
        self.complexity = 3
        self.n_cosmological = 2
        self.n_nuisance = 6
        self.name = 'Damped-driven harmonic oscillator with position-dependent gravity cosmology'
        self.n_parameters =  self.n_nuisance + self.n_cosmological
        self.fiducial_cosmological_parameters = np.array([1.0, .25]) # w
        self.cosmological_parameter_names = ['frequency of osciallator', 'height-dependence of oscillator frequency']
        self.fiducial_nuisance_parameters = np.array([0.50,0.20,0.3,np.pi,.0,1.0])
        self.nuisance_parameter_names = ['drag','driver amplitude','driver frequency','driver phase','oscillator initial position','oscillator initial velocity']

    def get_parameter_set(self):
        parameters = np.concatenate([self.fiducial_cosmological_parameters,self.fiducial_nuisance_parameters])
        return parameters

# copied from experiment.py
    def generate_model_data_vector(self, times, parameters = None):
        '''
        Generate ideal data vector from cosmology parameters, nuisance
        parameters, and experimental parameters
        '''
        w0 = parameters[0] # frequency of oscillator at midpoint
        wa = parameters[1] # change in frequency of oscillator with angle theta
        c = parameters[2] # drag coefficient
        A = parameters[3] # amplitude of driver
        wd = parameters[4] # frequency of driver
        phid = parameters[5] # phase of driver
        theta_x0 = parameters[6] # initial position of oscillator
        theta_v0 = parameters[7] # initial velocity of oscillator

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
            weff = w0 - x*wa # pendulum gets lighter as we go up
            up = (forcing_function(t) - c * u - weff**2 * x)
            return np.array([xp,up])

        # minimum and maximum times over which the solution should be calculated
        interval = (np.min(times),np.max(times))

        # solving the oscillator equations of motion for the total interval
        solution = solver(oscillator_eqns, interval, np.array([parameters[-2],parameters[-1]]), t_eval = times)
        self.ideal_data_vector = solution.y[0]
        return solution.y[0]

from abc import ABCMeta, abstractmethod
import numpy as np

class Cosmology(metaclass = ABCMeta):
    def __init__(self,complexity):
        self.n_parameters = False
        self.n_cosmological = False
        self.n_nuisance = False
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
        self.best_fit_cosmological_parameters = np.zeros(self.n_cosmological)

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
        self.best_fit_cosmological_parameters = np.zeros(self.n_cosmological)+1.0
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
        self.best_fit_cosmological_parameters = np.zeros(self.n_cosmological)+10.0

    def get_parameter_set(self):
         parameters = np.zeros(self.n_cosmological)+10.0

         return parameters

    def generate_model_data_vector(self,times,parameters):
#        self.best_fit_cosmological_parameters = get_parameter_set()
        model_data_vector = parameters[0]*np.ones(len(times)) + parameters[1]**2

        return model_data_vector

class StraightLineCosmology(Cosmology):
    def __init__(self,complexity):
        self.complexity = 0
        pass

    def generate_model_data_vector(self,times,parameters):
        model_data_vector = constant_theta_0 *\
        (constant_g/constant_l) * self.times + constant_phase

    def get_parameter_set(self):
        names = [par.name for par in parameters]

        constant_g = parameters[ names.index('constant_g') ].value
        constant_l = parameters[ names.index('constant_l') ].value
        constant_theta_0 = parameters[ names.index('constant_theta_0')].value
        constant_phase = parameters[ names.index('constant_phase') ].value


class CosineCosmology(Cosmology):
    def __init__(self,complexity):
        pass

    def generate_model_data_vector(self,):
        # define model for data with parameters above
        model_data_vector = constant_theta_0 \
        * np.cos(np.sqrt(constant_g /constant_l) * self.times + constant_phase)


    def get_parameter_set(self):
        names = [par.name for par in parameters]

        constant_g = parameters[ names.index('constant_g') ].value
        constant_l = parameters[ names.index('constant_l') ].value
        constant_theta_0 = parameters[ names.index('constant_theta_0')].value
        constant_phase = parameters[ names.index('constant_phase') ].value


class TrueCosmology(Cosmology):
    def __init__(self,complexity):
        self.n_cosmological = 2
        self.n_nuisance = 6
        self.n_parameters =  self.n_nuisance + self.n_cosmological

    def get_parameter_set(self):
        return np.zeros(self.n_parameters)

# copied from experiment.py
    def generate_model_data_vector(self, times, parameters = None):
        '''
        Generate ideal data vector from cosmology parameters, nuisance
        parameters, and experimental parameters
        '''
        g = parameters[0]
        l = parameters[1]

        c = parameters[2]
        A = parameters[3]
        wd = parameters[4]
        phid = parameters[5]
        theta_x0 = parameters[6]
        theta_v0 = parameters[7]

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
            up = (forcing_function(t) - c * u - l * x) / g
            return np.array([xp,up])

        # minimum and maximum times over which the solution should be calculated
        interval = (np.min(times),np.max(times))

        # solving the oscillator equations of motion for the total interval
        solution = solver(oscillator_eqns, interval, np.array([parameters[-2],parameters[-1]]), t_eval = times)
        self.ideal_data_vector = solution.y[0]
        return solution.y[0]

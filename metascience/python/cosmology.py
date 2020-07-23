from abc import ABCMeta, abstractmethod
import numpy as np

class Cosmology(metaclass = ABCMeta):
    def __init__(self,complexity):
        self.n_parameters = False
        pass

    @abstractmethod
    def get_parameter_set(self):
        pass

    @abstractmethod
    def generate_model_data_vector(self,):
        pass



class CargoCultCosmology(Cosmology):
    def __init__(self,complexity):
        self.complexity = 0
        self.n_parameters = 2
        self.best_fit_cosmological_parameters = np.zeros(n_parameters)
        self.best_fit_cosmological_parameter_covariance=np.eye(n_parameters)

    def get_parameter_set(self):
         parameters = np.zeros(self.n_parameters)
         return parameters


    def generate_model_data_vector(self,parameters):
        self.best_fit_cosmological_parameters = get_parameter_set()
        self.best_fit_cosmological_parameter_covariance = np.eye(self.n_parameters)

        

        model_data_vector = self.best_fit_cosmological_parameters[0]\
        *np.ones(len(self.times))**2 + self.best_fit_cosmological_parameters[1]

class CargoCultCosmology_Ones(Cosmology):
    def __init__(self,complexity):
        self.complexity = 0
        self.best_fit_cosmological_parameters = np.zeros(n_parameters)
        self.best_fit_cosmological_parameter_covariance=np.eye(n_parameters)
#        self.chi2 = 0.

    def get_parameter_set(self):
         n_parameters = 2
         parameters = np.zeros(n_parameters)+1.0
         self.best_fit_cosmological_parameters = np.zeros(n_parameters)+1.0

    def generate_model_data_vector(self):
        self.best_fit_cosmological_parameters = get_parameter_set()
#        self.chi2 = n_parameters
        self.best_fit_cosmological_parameter_covariance = np.eye(n_parameters)

        model_data_vector = self.best_fit_cosmological_parameters[0]\
        *np.ones(len(self.times)) + self.best_fit_cosmological_parameters[1]


class CargoCultCosmology_Tens(Cosmology):
    def __init__(self,complexity):
        self.complexity = 0
        self.best_fit_cosmological_parameters = np.zeros(n_parameters)
        self.best_fit_cosmological_parameter_covariance=np.eye(n_parameters)
#        self.chi2 = 0.

    def get_parameter_set(self):
         n_parameters = 2
         parameters = np.zeros(n_parameters)+10.0
         self.best_fit_cosmological_parameters = np.zeros(n_parameters)+10.0

    def generate_model_data_vector(self):
        self.best_fit_cosmological_parameters = get_parameter_set()
#        self.chi2 = n_parameters
        self.best_fit_cosmological_parameter_covariance = np.eye(n_parameters)

        model_data_vector = self.best_fit_cosmological_parameters[0]\
        *np.ones(len(self.times)) + self.best_fit_cosmological_parameters[1]**2


class StraightLineCosmology(Cosmology):
    def __init__(self,complexity):
        self.complexity = 0
        pass

    def generate_model_data_vector(self,):
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

# copied from experiment.py
        def _get_ideal_data_vector(self):
            '''
            Generate ideal data vector from cosmology parameters, nuisance
            parameters, and experimental parameters
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
            interval = (np.min(self.times),np.max(self.times))

            # solving the oscillator equations of motion for the total interval
            solution = solver(oscillator_eqns, interval, np.array([self.constant_theta_0, self.constant_theta_v0]), t_eval = self.times)
            self.ideal_data_vector = solution.y[0]
            return solution.y[0]

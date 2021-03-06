from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
#import cosmology ???
import ipdb
import matplotlib.pyplot as plt

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
        self.lowest_systematics_coeff = 500
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
        self.systematics_only = np.zeros_like(data_vector)
        for nu,coeff in enumerate(parameters):
            arg = self.times / (np.max(self.times) - np.min(self.times))
            arg = arg - np.min(arg)
            actual_nu = 2*nu+1+self.lowest_systematics_coeff
            #print(f"nu: {actual_nu}")
            thissys = coeff*scipy.special.eval_laguerre(2*nu+1+self.lowest_systematics_coeff,arg)
            #thissys = coeff*scipy.special.hankel1(nu,self.times/np.max(self.times)*2*np.pi)
            if np.sum(~np.isfinite(thissys)) > 0:
                if nu == 0:
                    thissys[~np.isfinite(thissys)] = coeff
                else:
                    thissys[~np.isfinite(thissys)] = 0.
            self.systematics_only = self.systematics_only+thissys.real
            data_vector = data_vector + thissys.real

        return data_vector

    def fit_model(self):
        '''
        Fit a model to generate a posterior of best-fit parameters
        notes:
            maybe we need an explicit jacobian because the model has cosine in it?
        '''

        def evaluate_logL(parameters,return_chisq = False):
            '''
            What can we infer about the world from the provided experiment?
            Takes a cosmology, returns a log-posterior or similar.
            notes:
                is the syntax right for something dotted with itself?
                is the covariance attached in the right way?
                let's try including a prior over the systematics...
            '''
            model_data_vector = self.cosmology.generate_model_data_vector(self.times,parameters[:self.cosmology.n_parameters])
            #print(parameters)
            # add systematics to the model
            model_with_systematics = self._add_systematics(model_data_vector,parameters = parameters[self.cosmology.n_parameters:])
            # calculate difference between model and data
            delta = model_with_systematics - self.measured_data_vector

            # calcualte chisq

            chisq = np.dot(delta.T, np.dot(self.inverse_covariance,delta))
            chi = np.dot(self.inverse_covariance,delta)

            syspars = parameters[self.cosmology.n_parameters:]
            syspars_prior = .0001/(np.arange(len(syspars))+1)
            chi_with_prior = np.concatenate([chi,syspars/syspars_prior])

            if return_chisq:
                return chisq
            else:
                return chi#_with_prior

        def _Fisher_covariance(log_likelihood, parameters, step_size = 0.01,check=False):

            def eval_model(parameters):
                model_data_vector = self.cosmology.generate_model_data_vector(self.times,parameters[:self.cosmology.n_parameters])
                model_with_systematics = self._add_systematics(model_data_vector,parameters = parameters[self.cosmology.n_parameters:])
                return model_with_systematics

            param_derivs = np.zeros((self.times.size,parameters.size)) # Derivatives of the model wrt to the parameters
            for iparam in range(parameters.size):
                param_p = parameters.copy()
                param_p[iparam] = param_p[iparam] + step_size * parameters[iparam]
                dmp = eval_model(param_p)
                param_m = parameters.copy()
                param_m[iparam] = param_m[iparam] - step_size * parameters[iparam]
                dmm = eval_model(param_m)
                param_derivs[:,iparam] = (dmp - dmm) / (2*step_size*parameters[iparam])

            Fisher = np.zeros((parameters.size,parameters.size))
            for iparam in range(parameters.size):
                for jparam in range(parameters.size):
                    Fisher[iparam,jparam] = param_derivs[:,iparam].dot(self.inverse_covariance).dot(param_derivs[:,jparam])
            C = np.linalg.inv(Fisher)
            if check:
                npts = 250
                param_range = np.linspace( parameters[0] - 5*np.sqrt(1./Fisher[0,0]), parameters[0] + 5*np.sqrt(1./Fisher[0,0]), npts )
                L = np.zeros(npts)
                for i in range(npts):
                    this_params = parameters.copy()
                    this_params[0] = param_range[i]
                    L[i] = evaluate_logL(this_params,return_chisq=True)

                fig,ax = plt.subplots()
                best_param = param_range[np.argmin(L)]
                ax.plot(param_range,np.exp(-L + np.min(L) )*1./np.sqrt(2*np.pi*1./Fisher[0,0]))
                ax.axvline(parameters[0] - np.sqrt(1./Fisher[0,0]),color='red',linestyle='--')
                ax.axvline(best_param,color='black',linestyle='--',label='maxL param')
                ax.axvline(parameters[0],color='cyan',linestyle='--',label='best-fit param')
                ax.axvline(parameters[0] + np.sqrt(1./Fisher[0,0]),color='red',linestyle='--')
                ax.set_xlabel('parameter 0')
                ax.set_ylabel('likelihood')
                ax.legend(loc='best')
                plt.savefig('likelihood_check_plot.png')
                plt.close(fig)
            return C

        # generate a guess in the right order.
        guess = np.concatenate([self.cosmology.fiducial_cosmological_parameters,self.cosmology.fiducial_nuisance_parameters,self.starting_systematics_parameters])
        print(f"Initial guess:{guess}")
        # fit for parameters
        # Think about the number of iterations required to make convergence likely under our normal modeling scenarios.
        best_fit_parameters = scipy.optimize.root(evaluate_logL, guess, method = 'lm', options = {'maxiter':10000*(guess.size+1)})
        self.chi2 = evaluate_logL(best_fit_parameters.x,return_chisq=True)#/len(self.measured_data_vector) # save the best-fit chi2
        # This is what the interpreter thinks the data would look like without systematics, based on its best fit
        self.best_fit_ideal_model = self.cosmology.generate_model_data_vector(self.times,best_fit_parameters.x[:self.cosmology.n_parameters])
        # This is what the interpreter's best-fit model thinks the data should look like, with systematics.
        self.best_fit_observed_model = self._add_systematics(self.best_fit_ideal_model,parameters = best_fit_parameters.x[self.cosmology.n_parameters:])
        # generate list of paraemters for each kind of parameter in the correct order
        # ..todo: make this consistent with new parameters definition.
        self.best_fit_cosmological_parameters = best_fit_parameters.x[:self.cosmology.n_cosmological]
        self.best_fit_nuisance_parameters = best_fit_parameters.x[self.cosmology.n_cosmological:self.cosmology.n_parameters]
        self.best_fit_systematics_parameters = best_fit_parameters.x[self.cosmology.n_parameters:]

        print(f"best fit:{best_fit_parameters.x}")
        # Find and store the best-fit parameter covariance.
        # Note that cov_x isn't the true parameter covariance, it is a relative covariance,
        # So our covmat needs to be rescaled by the residuals aka chi2
        # See https://scipy.github.io/devdocs/generated/scipy.optimize.curve_fit.html under absolute_sigma parameter
        # pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)
        try:
            relCovmat = best_fit_parameters.cov_x[:self.cosmology.n_cosmological,:self.cosmology.n_cosmological]
            absCovmat = relCovmat*(len(self.times)-self.cosmology.n_cosmological)/self.chi2
            self.best_fit_cosmological_parameter_covariance = absCovmat
        except:
            self.best_fit_cosmological_parameter_covariance = -1*np.eye(self.cosmology.n_cosmological)
            print(f"fit failed to coverge because: {best_fit_parameters.message} ")
            print('covariance is None: setting to -1')
        # removing the Fisher for now to test the consensu module that blows up errors
        fullcov = _Fisher_covariance(evaluate_logL,best_fit_parameters.x,check=True)
        #print(fullcov[:self.cosmology.n_cosmological,:self.cosmology.n_cosmological])
        #print('huh covariance')
        #print(self.best_fit_cosmological_parameter_covariance)
        # Uncomment below to use Fisher covariance
        self.best_fit_cosmological_parameter_covariance = fullcov[:self.cosmology.n_cosmological,:self.cosmology.n_cosmological]
        # apply success flag from fit parameters to fit status
        self.fit_status = best_fit_parameters.success




    def elaborate_systematics(self):

        # some conditional logic here
        # then:
        def _add_systematics():
            # as a new kind of systematics function
            pass

        pass

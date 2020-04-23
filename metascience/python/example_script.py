import matplotlib.pyplot as pyplot
import experiment
import interpret

# Choose a set of parameters

cosmology_parameters = {}
cosmology_parameters['constant_g'] = 9.8

nuisance_parameters = {}
nuisance_parameters['constant_l'] = 2.0
nuisance_parameters['constant_theta_0'] = 0.1
nuisance_parameters['constant_theta_v0'] =0.1


noise_parameters = {}
noise_parameters['noise_std_dev'] = 0.5

systematics_parameters = {}
systematics_parameters['boost_deriv']  = 5

experimental_parameters = {}
experimental_parameters['time_between_measurements'] = 0.1
experimental_parameters['number_of_measurements'] = 10

seed=999


# Create an instance of the Experiment class:

myPendulum = experiment.SimplePendulumExperiment(cosmology_parameters = None,
                                                 nuisance_parameters = None,
                                                 experimental_parameters = None,
                                                 noise_parameters = None,
                                                 systematics_parameters = None,
                                                 seed = 999)

# Get a data vector.
myPendulum.generate_data()
data = myPendulum.data_vector
# Maybe do some plotting?
plt.plot(data)
plt.show()


# Now we want to try interpreting the results of the experiment.
# once we've run myPendulum.generate_data()
# Define a bunch of new parameters that will govern the experiment.
interpreter_cosmology_parameters = {}
interpreter_cosmology_parameters['constant_g'] = 9.8

interpreter_nuisance_parameters = {}
interpreter_nuisance_parameters['constant_l'] = 1.0
interpreter_nuisance_parameters['constant_theta_0'] =  0.0
interpreter_nuisance_parameters['constant_theta_v0'] = 0.0

interpreter_noise_parameters = {}
interpreter_noise_parameters['noise_std_dev'] = 0.15

interpreter_systematics_parameters = {}
interpreter_systematics_parameters['function']  = 'hankel'
interpreter_systematics_parameters['coeff'] = np.zeros(2)

myPendulumIntepreter = interpret.SimplePendulumExperimentInterpreter(experiment = None,
                                    cosmology_parameters = interpreter_cosmology_parameters,
                                    nuisance_parameters = interpreter_nuisance_parameters,
                                    noise_parameters = interpreter_noise_parameters,
                                    systematics_parameters = interpreter_systematics_parameters )

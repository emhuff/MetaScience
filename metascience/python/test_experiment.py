import cosmology
import experiment
import numpy as np
import matplotlib.pyplot as plt


'''
truth_parameters[0] = 9.8 # g cosmological parameter 1
truth_parameters[1] = 1.0 # l cosmological parameter 2
truth_parameters[2] = 0.01 # C -- drag/damping coefficient
truth_parameters[3] = .0 # A -- amplitude of forcing function
truth_parameters[4] = .1 # wd -- freq. of forcing function
truth_parameters[5] = 0.0 # phid -- phase of forcing function
truth_parameters[6] = 1.0  # theta_x0 -- initial position of pendulum
truth_parameters[7] = 0.0 # theta_v0 -- initial velocity of pendulum
'''

true_cosmology_parameters = np.array([9.8, 1])
true_nuisance_parameters = np.array([0.01, .0, .1, 0.0, 1.0, 1.])
true_systematics_parameters = np.array([1.])
experimental_parameters = {'times':np.linspace(0,10,500)}
noise_parameters = np.array([0.1])
truth = cosmology.TrueCosmology()

experiment = experiment.SimplePendulumExperiment(cosmology_parameters = true_cosmology_parameters,
                                                nuisance_parameters = true_nuisance_parameters,
                                                experimental_parameters = experimental_parameters,
                                                cosmology = truth,
                                                noise_parameters = noise_parameters,
                                                systematics_parameters = true_systematics_parameters)
experiment.generate_data()
plt.plot(experiment.times,experiment.observed_data_vector,label='observed')
plt.plot(experiment.times,experiment.ideal_data_vector,label='observed')
plt.show()

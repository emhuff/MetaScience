import numpy as np
import matplotlib.pyplot as plt
import experiment
import interpret
import cosmology

truth = cosmology.TrueCosmology()
experimental_parameters = {'times':np.linspace(0,5,500)}
noise_parameters = np.array([0.01])
true_systematics_parameters = np.array([.01])
true_parameters = truth.get_parameter_set()
true_parameters[3] = np.sqrt(12.)
true_parameters[1] = 0.5
pendulum = experiment.SimplePendulumExperiment(cosmology=truth,
                                               experimental_parameters=experimental_parameters,
                                               cosmology_parameters=true_parameters[:truth.n_cosmological],
                                               nuisance_parameters=true_parameters[truth.n_cosmological:],
                                               systematics_parameters=true_systematics_parameters,
                                               noise_parameters = noise_parameters)
pendulum.generate_data()

model = cosmology.CosineCosmology()
#noise_parameters = np.array([0.2])
starting_systematics_parameters = np.array([0.,])

pendulum_interp = interpret.SimplePendulumExperimentInterpreter(experiment = pendulum,
                                                                 starting_systematics_parameters = starting_systematics_parameters,
                                                                 noise_parameters = noise_parameters,
                                                                 prior = None,
                                                                 cosmology = model)
pendulum_interp.fit_model()
print(f"fit status: {pendulum_interp.fit_status}")
w_true = truth.fiducial_cosmological_parameters[0]
w_fit = pendulum_interp.best_fit_cosmological_parameters[0]

print(f"best-fit cosmological parameters: {w_fit}")
print(f"true cosmological parameters: {w_true}")

plt.errorbar(pendulum.times,pendulum.observed_data_vector,np.zeros(pendulum.times.size)+noise_parameters,fmt='o',label='data',)
plt.plot(pendulum.times,pendulum.ideal_data_vector,label='true (ideal)')
plt.plot(pendulum.times,pendulum_interp.best_fit_observed_model,label='fit')
plt.plot(pendulum.times,pendulum_interp.best_fit_ideal_model,label='ideal fit')
plt.legend(loc='best')
plt.show()

# Show the systematics.

ipdb.set_trace()

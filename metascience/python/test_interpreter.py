import numpy as np
import matplotlib.pyplot as plt
import experiment
import interpret
import cosmology

truth = cosmology.ExponentialCosmology()
experimental_parameters = {'times':np.linspace(0,7,500)}
noise_parameters = np.array([.1])
n_sys_coeff = 5
true_systematics_parameters = .1/(np.arange(n_sys_coeff)+1) * np.random.randn(n_sys_coeff)
true_parameters = truth.get_parameter_set()
pendulum = experiment.SimplePendulumExperiment(cosmology=truth,
                                               experimental_parameters=experimental_parameters,
                                               cosmology_parameters=true_parameters[:truth.n_cosmological],
                                               nuisance_parameters=true_parameters[truth.n_cosmological:],
                                               systematics_parameters=true_systematics_parameters,
                                               noise_parameters = noise_parameters)
pendulum.generate_data()

model = cosmology.DampedDrivenOscillatorVariableGCosmology()
#noise_parameters = np.array([0.2])
starting_systematics_parameters = np.zeros_like(true_systematics_parameters)

pendulum_interp = interpret.SimplePendulumExperimentInterpreter(experiment = pendulum,
                                                                 starting_systematics_parameters = starting_systematics_parameters,
                                                                 noise_parameters = noise_parameters,
                                                                 prior = None,
                                                                 cosmology = model)

pendulum_interp.fit_model()
print(f"fit status: {pendulum_interp.fit_status}")
w_true = truth.fiducial_cosmological_parameters[0]
w_fit = pendulum_interp.best_fit_cosmological_parameters[0]

chi2dof = pendulum_interp.chi2*1./len(pendulum.observed_data_vector)
print(f"best-fit cosmological parameters: {w_fit}")
print(f"true cosmological parameters: {w_true}")
print(f"goodness-of-fit: {chi2dof}")


plt.errorbar(pendulum.times,pendulum.observed_data_vector,np.zeros(pendulum.times.size)+noise_parameters,fmt=',',linestyle='None', label='data',zorder=1)
plt.plot(pendulum.times,pendulum_interp.best_fit_observed_model,label='fit',zorder=2,linewidth=3)
plt.plot(pendulum.times,pendulum_interp.best_fit_ideal_model,label='ideal fit',zorder=2)
plt.plot(pendulum.times,pendulum.ideal_data_vector,'-.',label='true (ideal)',zorder=3)

plt.legend(loc='best')
plt.show()
ipdb.set_trace()
# Show the systematics.

import numpy as np
import matplotlib.pyplot as plt
import experiment
import interpret
import cosmology

truth = cosmology.TrueCosmology()
experimental_parameters = {'times':np.linspace(0,10,500)}
noise_parameters = np.array([0.1])
true_systematics_parameters = np.array([1.])
pendulum = experiment.SimplePendulumExperiment(cosmology=truth,
                                               experimental_parameters=experimental_parameters,
                                               cosmology_parameters=truth.best_fit_cosmological_parameters,
                                               nuisance_parameters=truth.best_fit_nuisance_parameters,
                                               systematics_parameters=true_systematics_parameters,
                                               noise_parameters = noise_parameters)
pendulum.generate_data()

model = cosmology.CosineCosmology()
noise_parameters = np.array([0.1])
starting_systematics_parameters = np.array([0.,0.,0.])

pendulum_interp = interpret.SimplePendulumExperimentInterpreter(experiment = pendulum,
                                                                 starting_systematics_parameters = starting_systematics_parameters,
                                                                 noise_parameters = noise_parameters,
                                                                 prior = None,
                                                                 cosmology = model)
pendulum_interp.fit_model()
print(f"fit status: {pendulum_interp.fit_status}")
print(f"best-fit cosmological parameters: {pendulum_interp.best_fit_cosmological_parameters}")

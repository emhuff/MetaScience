import numpy as np
import matplotlib.pyplot as plt
import experiment
import interpret
import cosmology
import consensus

truth = cosmology.TrueCosmology()
# These parameters are shared:
true_parameters = truth.get_parameter_set()
true_parameters[3] = np.sqrt(12.)
true_parameters[1] = 0.5
# We need two different experiments.
experimental_parameters1 = {'times':np.linspace(2.,8.,500)}
noise_parameters1 = np.array([0.03])
true_systematics_parameters1 = np.array([.00])
pendulum1 = experiment.SimplePendulumExperiment(cosmology=truth,
                                               experimental_parameters=experimental_parameters1,
                                               cosmology_parameters=true_parameters[:truth.n_cosmological],
                                               nuisance_parameters=true_parameters[truth.n_cosmological:],
                                               systematics_parameters=true_systematics_parameters1,
                                               noise_parameters = noise_parameters1,seed=110)
pendulum1.generate_data()


experimental_parameters2 = {'times':np.linspace(0,10,500)}
noise_parameters2 = np.array([0.03])
true_systematics_parameters2 = np.array([.00])
pendulum2 = experiment.SimplePendulumExperiment(cosmology=truth,
                                               experimental_parameters=experimental_parameters2,
                                               cosmology_parameters=true_parameters[:truth.n_cosmological],
                                               nuisance_parameters=true_parameters[truth.n_cosmological:],
                                               systematics_parameters=true_systematics_parameters2,
                                               noise_parameters = noise_parameters2, seed = 111)
pendulum2.generate_data()

#plt.plot(pendulum1.times,pendulum1.observed_data_vector,label='1')
#plt.plot(pendulum2.times,pendulum2.observed_data_vector,label='2')
#plt.show()

n_experiments = 2
experiments = [pendulum1,pendulum2]
noise_parameters = [noise_parameters1,noise_parameters2]

cosmologies = [cosmology.TrueCosmology(),cosmology.CosineCosmology()]#, cosmology.ExponentialCosmology()] # we will start with the last item first!
this_cosmology= cosmologies.pop()
interpreters = []
n_systematics_parameters = [1,1]
starting_systematics_parameters = [np.zeros(i) for i in n_systematics_parameters]
systematics_parameters = starting_systematics_parameters

for i in range(n_experiments):
    print(noise_parameters[i], 'noise parameters for experiment ', i )
    interpreters.append(interpret.SimplePendulumExperimentInterpreter(experiment=experiments[i],
    cosmology=this_cosmology, starting_systematics_parameters=starting_systematics_parameters[i], noise_parameters=noise_parameters[i]))
#interpreters.append(interpret.SimplePendulumExperimentInterpreter(experiment = pendulum1, cosmology=this_cosmology, starting_systematics_parameters = starting_systematics_parameters[0], noise_parameters = noise_parameters1))
#interpreters.append(interpret.SimplePendulumExperimentInterpreter(experiment = pendulum2, cosmology=this_cosmology, starting_systematics_parameters = starting_systematics_parameters[1], noise_parameters = noise_parameters2))



n_iter = 100
still_ok = True
systematics_iter = np.zeros(n_experiments)
for iter in range(n_iter):
    print(f"------------------------------")
    print(f"Iteration {iter}:")
    print(f"Using cosmology: {this_cosmology.name}")
    for interpreter in interpreters:
        interpreter.fit_model()
        if np.any(np.diag(interpreter.best_fit_cosmological_parameter_covariance) < 0):
            print(f"the fit didn't proceed, your errors are {np.diag(interpreter.best_fit_cosmological_parameter_covariance)}- you should probably check your data, soldier")
            still_ok = False
            break
        else:
            errors = np.sqrt(np.diag(interpreter.best_fit_cosmological_parameter_covariance))

        print(f"best-fit parameters: {interpreter.best_fit_cosmological_parameters}")
        print(f"best-fit parameter errors: {errors}")
        print(f"fit chi2: {interpreter.chi2}")
        print(f"fit chi2/dof: {interpreter.chi2/interpreter.measured_data_vector.size}")

    if still_ok == False: break

    # Plot the fits.
    filename = f"pendulum_iter-{iter:03}.png"
    fig,ax = plt.subplots(figsize=(7,7))
    ax.plot(pendulum1.times,pendulum1.observed_data_vector,label='data 1',marker='.')
    ax.plot(interpreters[0].times,interpreters[0].best_fit_observed_model,label='model 1')
    ax.plot(interpreters[0].times,interpreters[0].best_fit_ideal_model,label='ideal 1',linestyle='--')
    ax.plot(pendulum2.times,pendulum2.observed_data_vector,label='data 2',marker='.')
    ax.plot(interpreters[1].times,interpreters[1].best_fit_observed_model,label='model 2')
    ax.plot(interpreters[1].times,interpreters[1].best_fit_ideal_model,label='ideal 2',linestyle='--')
    ax.legend(loc='best')
    fig.savefig(filename)
    plt.close(fig)

    # Now pass the result to the consensus.
    impatient = consensus.ImpatientConsensus(interpretations = interpreters)
    impatient.tension_metric()
    print(f"value of the tension parameter: {impatient.tm}")
    print(f"tension: {impatient.is_tension}")
    impatient.render_judgment(number_of_tries = np.max(systematics_iter))
    if np.max(systematics_iter) > impatient.patience:
        print(f"Got tired of further refining experiments after {impatient.patience} iterations. Changing the cosmology")
    if (not impatient.is_tension) and (np.sum(impatient.systematics_judgment) == 0):
        print('No tension, and everybody fits the data yay!')
        break
    if impatient.cosmology_judgment is True:
        print(f"Updating the cosmology")
        systematics_iter[:] = 0
        if len(cosmologies) == 0:
            print('Ran out of cosmologies to try!')
            break
        this_cosmology = cosmologies.pop()
        for i,interpreter in enumerate(interpreters):
            interpreters[i] = interpret.SimplePendulumExperimentInterpreter(experiment = experiments[i], cosmology=this_cosmology,
            starting_systematics_parameters = starting_systematics_parameters[i], noise_parameters = noise_parameters[i])

# here we have some choice about whether to start from square 1 with systematics parameters (if not, could try interpreter.best_fit_systematics_parameters)
    else:
        if np.sum(impatient.systematics_judgment) > 0:
            systematics_iter[impatient.systematics_judgment] = systematics_iter[impatient.systematics_judgment]+1
            for i,this_judgment in enumerate(impatient.systematics_judgment):
                if this_judgment:
                    print(f"Adding systematic error sophistication to interpreter {i}.")
                    systematics_parameters[i] = np.concatenate((systematics_parameters[i],np.zeros(1)))
                    interpreters[i] = interpret.SimplePendulumExperimentInterpreter(experiment = experiments[i], cosmology=this_cosmology,
                    starting_systematics_parameters = systematics_parameters[i], noise_parameters = noise_parameters[i])



#
# Now the loop:
#  1. Estimate parameters.
#  2. Initialize a consensus using all interpreters
#  3. Estimate tension
#  4. Render judgment
#  5. conditional logic from judgment:
#      5a. for one or more experiments: extend the starting systematics vector, re-initialize interpreter
#      5b. for all experiments: re-initialize with a new cosmology

# for the future:
#      5b. Change how one of the experiments is collecting data
#  6. (maybe: generate a new data from the experiments, re-initialize interpreters with above choices and new experiments)

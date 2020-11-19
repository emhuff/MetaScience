import numpy as np
import matplotlib.pyplot as plt
import experiment
import interpret
import cosmology
import consensus
import pickle
import copy
import ipdb

# From inifile, read in:
# - name of consensus choices, number determined by length
# - names of experiments and interpreters (read until done, take length of vector to be the number of names)
# - list of cosmologies
# - true cosmology
# - starting systematics parameters?
# - true systematics parameters?
# - impatient value (defaults to 10)
#
# Specify how many times you want to run the simulation, set list of random seeds of that length
# For rand in randomSeedList:
# for each consensus in list:
# run consensus record the final results
# --> (tension_metric, final parameter chi2, final data chi2 for all interpreters, number of iterations it took)
# save to file

class Results():
    def __init__(self):
        pass

class Configuration():
    def __init__(self):
        pass


## Fix scope issues! Variables defined here are in same namespace as those in the top-level function.

def run_consensus_compare(consensus_name, experiment_names, interpreter_names, interpreter_cosmologies,
                          true_cosmology, experimental_parameters, noise_parameters,max_iter = 1000,
                          number_of_systematics=1, true_systematics = np.array(0.0)):
    number_of_experiments = len(experiment_names)
    truth = getattr(cosmology, true_cosmology)()
    true_parameters = truth.get_parameter_set()
    true_systematics_parameters = [true_systematics]*number_of_experiments

    if true_cosmology=='DampedDrivenOscillatorCosmology':
        true_parameters[3] = np.sqrt(12.)
        true_parameters[1] = 0.5
    else:
        print(f'Ok buddy relax')

    assert number_of_experiments == len(interpreter_names), "Experiment_names and interpreter_names should be the same length."

    experiments = []
    interpreters = []
    this_cosmology = interpreter_cosmologies.pop()
    n_systematics_parameters = [number_of_systematics for i in interpreter_names]
    starting_systematics_parameters = [np.zeros(i) for i in n_systematics_parameters]
    systematics_parameters =  starting_systematics_parameters

    for i in range(number_of_experiments):
        experiments.append(getattr(experiment,experiment_names[i])(cosmology=truth,experimental_parameters=experimental_parameters[i],
                                                                   cosmology_parameters=true_parameters[:truth.n_cosmological],
                                                                   nuisance_parameters=true_parameters[truth.n_cosmological:],
                                                                   systematics_parameters=true_systematics_parameters[i],
                                                                   noise_parameters = noise_parameters[i],seed=110))
        # Make an interpreter for each experiment, initialized to default starting values.
        experiments[i].generate_data()
        interpreters.append(getattr(interpret,interpreter_names[i])(experiment=experiments[i],cosmology=this_cosmology,
                                                                    starting_systematics_parameters=starting_systematics_parameters[i],
                                                                    noise_parameters=noise_parameters[i]))
    # Run sequence.
    still_ok = True
    systematics_iter = np.zeros(number_of_experiments)
    converged = False
    for iter in range(max_iter):
        print(f"------------------------------")
        print(f"Iteration {iter}:")
        print(f"Using cosmology: {this_cosmology.name}")
        # Fit the models.
        for interpreter in interpreters:
            interpreter.fit_model()
            if np.any(np.diag(interpreter.best_fit_cosmological_parameter_covariance) < 0):
                print(f"the fit didn't proceed, your errors are {np.diag(interpreter.best_fit_cosmological_parameter_covariance)}- you should probably check your data, soldier")
                still_ok = False
                # TODO: fail more gracefully when the fitter doesn't converge.
                break
            else:
                errors = np.sqrt(np.diag(interpreter.best_fit_cosmological_parameter_covariance))

            print(f"best-fit parameters: {interpreter.best_fit_cosmological_parameters}")
            print(f"best-fit parameter errors: {errors}")
            print(f"fit chi2: {interpreter.chi2}")
            print(f"fit chi2/dof: {interpreter.chi2/interpreter.measured_data_vector.size}")

        if still_ok == False: break
        # Separate plotting function.
        this_consensus = getattr(consensus,consensus_name)(interpretations = interpreters, patience = 5)
        this_consensus.tension_metric()
        print(f"value of the tension parameter: {this_consensus.tm}")
        print(f"tension: {this_consensus.is_tension}")
        this_consensus.render_judgment(number_of_tries = np.max(systematics_iter))

        if np.max(systematics_iter) > this_consensus.patience:
            print(f"Got tired of further refining experiments after {this_consensus.patience} iterations. Changing the cosmology")
        if (not this_consensus.is_tension) and (np.sum(this_consensus.systematics_judgment) == 0):
            print('No tension, and everybody fits the data yay!')
            converged = True
            break
        if this_consensus.cosmology_judgment is True:
            print(f"Updating the cosmology")
            systematics_iter[:] = 0
            if len(interpreter_cosmologies) == 0:
                print('Ran out of cosmologies to try!')
                break
            this_cosmology = interpreter_cosmologies.pop()
            for i,interpreter in enumerate(interpreters):
                starting_systematics_parameters = [np.zeros(i) for i in n_systematics_parameters] # note: not sure why we need to re-define but we do!
                interpreters[i] = getattr(interpret,interpreter_names[i])(experiment = experiments[i], cosmology=this_cosmology,
                                                                          starting_systematics_parameters = starting_systematics_parameters[i],
                                                                          noise_parameters = noise_parameters[i])
        else:
            if np.sum(this_consensus.systematics_judgment) > 0:
                systematics_iter[this_consensus.systematics_judgment] = systematics_iter[this_consensus.systematics_judgment]+1
                for i,this_judgment in enumerate(this_consensus.systematics_judgment):
                    if this_judgment:
                        print(f"Adding systematic error sophistication to interpreter {i}.")
                        systematics_parameters[i] = np.concatenate((systematics_parameters[i],np.zeros(1)))
                        interpreters[i] = getattr(interpret,interpreter_names[i])(experiment = experiments[i], cosmology=this_cosmology,
                                                                                  starting_systematics_parameters = systematics_parameters[i],
                                                                                  noise_parameters = noise_parameters[i])

    print('out of loop')
    #
    '''
    Now what? Summarize what we know.
      1. Did we converge, or hit the max_iter limit?
      2. Did we land on the right cosmology, or not?
      3. How long did it take to finish?
      4. What are the posteriors for the final parameters (are the right? What's the tension with the true parameters? or similar)
    '''
    result = Results()
    result.consensus_name = consensus_name
    result.experiment_names = experiment_names
    result.true_cosmology = true_cosmology
    result.interpreter_names = interpreter_names
    result.interpreter_cosmologies = [thing.name for thing in interpreter_cosmologies]
    result.experimental_parameters = experimental_parameters
    result.noise_parameters = noise_parameters
    result.converged = converged
    result.iterations = iter
    result.final_cosmology = this_cosmology.name
    result.final_tension_metric = this_consensus.tm
    result.true_systematics = true_systematics
    result.consensus_cosmological_parameters = this_consensus.consensus_cosmological_parameters
    result.cosmological_parameter_names = this_cosmology.cosmological_parameter_names
    #result.nuisance_parameters = this_consensus.best_fit_nuisance_parameters
    #result.nuisance_parameter_names = this_cosmology.nuisance_parameter_names
    result.consensus_parameter_covariance = this_consensus.consensus_parameter_covariance

    return result



consensusize = ['ImpatientConsensus', 'MostlyBetOnMeConsensus']

experimental_parameters=[{'times':np.linspace(2.,8.,500)},{'times':np.linspace(0,10,500)}]
noise_parameters = [np.array([0.03]), np.array([0.1])]
n_true_sys = 5
true_systematics = np.array([0.]) # 1./(np.arange(n_true_sys)+1)**2 * np.random.randn(n_true_sys)
number_of_consensus = len(consensusize)

experiment_names = ['SimplePendulumExperiment', 'SimplePendulumExperiment']
interpreter_names = ['SimplePendulumExperimentInterpreter','SimplePendulumExperimentInterpreter']
number_of_interpreters=len(interpreter_names)
interpreter_cosmologies = [cosmology.DampedDrivenOscillatorVariableGCosmology(), cosmology.DampedDrivenOscillatorCosmology(),
               cosmology.GaussianCosmology(),cosmology.BesselJCosmology(), cosmology.AiryCosmology(),
               cosmology.CosineCosmology(),cosmology.StraightLineCosmology()]

interpreter_cosmologies = [cosmology.DampedDrivenOscillatorVariableGCosmology(), cosmology.CosineCosmology(),cosmology.StraightLineCosmology()]

#true_cosmology = 'DampedDrivenOscillatorCosmology'
true_cosmology = 'CosineCosmology'

# TODO: wrap this in a loop that stores and (maybe?) visualizes results.

for this_consensus in consensusize:

    result = run_consensus_compare(consensusize[0], experiment_names, interpreter_names, interpreter_cosmologies, true_cosmology, experimental_parameters, noise_parameters, true_systematics = true_systematics)
    results_file = f"{this_consensus}-results.pickle".replace(" ","_")
    with open(results_file, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

ipdb.set_trace()

# To read:
'''
with open('data.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)
'''

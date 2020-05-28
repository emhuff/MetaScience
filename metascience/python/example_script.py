import matplotlib.pyplot as pyplot
import experiment
import interpret
#import prior
#import consensus
import matplotlib.pyplot as plt
import numpy as np
# Choose a set of parameters

cosmology_parameters = {}
cosmology_parameters['constant_g'] = 1000.0

nuisance_parameters = {}
nuisance_parameters['constant_l'] = 10.0
nuisance_parameters['constant_theta_0'] = 10.0
nuisance_parameters['constant_theta_v0'] =0.1


noise_parameters = {}
noise_parameters['noise_std_dev'] = 0.01

systematics_parameters = {}
systematics_parameters['boost_deriv']  = 0.2
systematics_parameters['drag_coeff'] = 0.3
systematics_parameters['driving_amp'] = 0.1
systematics_parameters['driving_freq'] = 2
systematics_parameters['driving_phase'] = 0.4

experimental_parameters = {}
experimental_parameters['time_between_measurements'] = .01
experimental_parameters['number_of_measurements'] = 500


pars
pars_experiment = []
seed = 999


# ==================================================================
# Create an instance of the Experiment class:
myPendulum = experiment.SimplePendulumExperiment(cosmology_parameters = cosmology_parameters,
                                                 nuisance_parameters = nuisance_parameters,
                                                 experimental_parameters = experimental_parameters,
                                                 noise_parameters = noise_parameters,
                                                 systematics_parameters = systematics_parameters,
                                                 seed = 999)

# Get a data vector.
myPendulum.generate_data()
ideal_data = myPendulum.ideal_data_vector
real_data =  myPendulum.data_vector
# Maybe do some plotting?
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
ax1.plot(myPendulum.times, ideal_data,label = 'without noise,sys.')
ax1.plot(myPendulum.times,real_data,label = 'with noise,sys.')
ax1.legend(loc = 'best')
ax1.set_ylabel('Theta')
ax1.set_xlabel('Time')
ax2.plot(myPendulum.times, real_data - ideal_data)
ax2.set_ylabel('residuals')
ax2.set_xlabel('Time')
plt.tight_layout()
plt.show()


# ==================================================================
# Now we want to try interpreting the results of the experiment.
# once we've run myPendulum.generate_data()
# Define a bunch of new parameters that will govern the experiment.
pars_interpreter_cosmology = Parameter(value=9.8, name='constant_g', lable='cosmology', description='gravitational constant')
par_interpreter_constant_phase = Parameter(value=0.0, name='constant_phase', label='nuisance', description='phase angle of pendulum swing')
par_interpreter_constant_theta_0 = Parameter(value=1.0, name='constant_theta_0', label='nuisance', description='offset angle of pendulum')
par_interpreter_constant_l = Parameter(value=1.0, name='constant_l', label='nuisance', description='length of pendulum')
par_interpreter_noise = Parameter(value=0.15, name='noise_std_dev', label='systematics', description='Standard deviation of the noise')
par_interpreter_henkel = Parameter(value=np.zeros(2), name='henkel', description='Henkel Function coefficients')
pars_interpreter = [par_interpreter_henkel, par_interpreter_noise, par_interpreter_constant_l,
    par_interpreter_constant_theta_0, par_interpreter_constant_phase, par_interpreter_cosmology]

# We will also need a prior in order to interpret the results.
# Mean of the prior:


myPendulumIntepreter = interpret.SimplePendulumExperimentInterpreter(experiment = myPendulum,
                                    cosmology_parameters = interpreter_cosmology_parameters,
                                    nuisance_parameters = interpreter_nuisance_parameters,
                                    systematics_parameters = interpreter_systematics_parameters,
                                    noise_parameters = interpreter_noise_parameters)

myPendulumIntepreter.fit_model()
print("cosmological parameters, best fit:")
print(myPendulum.best_fit_cosmological_parameters)
print("nuisance parameters, best fit:")
print(myPendulumIntepreter.best_fit_nuisance_parameters)

#this_consensus = consensus.AlwaysBetOnMeConsensus(interprations = [myPendulumIntepreter,myOtherPendulumInterpreter, etcInterpreter])

'''
Pseudocode for how this all works in the end.
1. Choose a set of true cosmological parameters.
2. Make 2 or more different experiment classes (corresponding to different experiments -- e.g., supernovae+CMB)
(2a. For pendulum - two different ways to get g? maybe that's JUST two similar experiments with slightly different nuiscance params (liks Lengths))
3. Set the true nuisance and systematics parameters for each
4. Make an interpretation object for each experiment, with **potentially different** values for all the parameters

consensus_parameters = []

for number_of_experiments:

    for i in range(n_iterations):
        for j in range(n_experiments):
            Generate data for each experiment.
            Fit the data from the experiment.

    # two ways to combine the outputs of the experiments
    Consensus = ConsensusClass([list of interpreters])

    # judgments based on the combined outputs of experiments
    judgments = Consensus.render_judgment()

    for judge in zip(judgments):
        experiment.update_model(judge)

    # there's one consensus approach
    consensus_parameters.append(Consensus.consensus_cosmological_parameters)




Do this again  but substitute in a different consensus
for each consensus approach:
    plot consensus_parameters - true_cosmological_parameters
'''

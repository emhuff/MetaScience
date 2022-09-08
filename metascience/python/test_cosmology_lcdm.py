import cosmology
import numpy as np
import matplotlib.pyplot as plt
import experiment
# Test the true cosmology.
truth = cosmology.LCDM_distanceModulus()

truth_parameters = truth.get_parameter_set()

truth_parameters[0]=70.
truth_parameters[1] = 0.26
truth_parameters[2] = 0.7
truth_parameters[3]  = 0.04

true_systematics=[np.array(0.0)]
noise_parameters=[np.array(0.03)]

# At what times are we generating the data?
redshifts = np.linspace(0.1,1.,500)
experimental_parameters={'redshifts':np.linspace(0.1,1.,50)}
# experimental_parameters['redshifts']=np.linspace(0.,1.,50)


# Now generate some data with this parameter set!
true_model = truth.generate_model_data_vector(redshifts,parameters=truth_parameters)


experiment_instance = experiment.DistanceModulusExperiment(cosmology=truth,experimental_parameters=experimental_parameters,
                                                            cosmology_parameters=truth_parameters[:truth.n_cosmological],
                                                            nuisance_parameters=truth_parameters[truth.n_cosmological:],
                                                            systematics_parameters=true_systematics,
                                                            noise_parameters = noise_parameters,seed=110)


data = experiment_instance.generate_data()

print(data) # TO DO: why is data 'None'?
print(huh)

# # Test the model cosmology.
# model =  cosmology.LCDM_distanceModulus()
# model_parameters = model.get_parameter_set()
# model_parameters = truth_parameters
# model_data = model.generate_model_data_vector(times,parameters=model_parameters)

# plot this.
plt.plot(redshifts,true_model,label='model')
plt.plot(experimental_parameters['redshifts'],data,label='data')
plt.xlabel('time')
plt.ylabel('position')
plt.legend(loc='best')
plt.show()

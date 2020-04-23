import matplotlib.pyplot as pyplot
import experiment
import interpret

# Choose a set of parameters

cosmology_parameters = {}
cosmology_parameters['constant_g']

nuisance_parameters = {}
nuisance_parameters = {'constant_l'}


noise_parameters = {}
systematics_parameters = {}
experimental_parameters = {}

seed=999
## etc... finish defining the necessary parameters


# Create an instance of the Experiment class:

myPendulum = experiment.SimplePendulumExperiment(cosmology_parameters = None,
                                                 nuisance_parameters = None,
                                                 experimental_parameters = None,
                                                 noise_parameters = None,
                                                 systematics_parameters = None,
                                                 seed=999)

# Get a data vector.
data = myPendulum.generate_data

# Maybe do some plotting?
plt.plot(data)
plt.show()

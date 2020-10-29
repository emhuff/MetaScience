import numpy as np
import matplotlib.pyplot as plt
import experiment
import interpret
import cosmology
import consensus

# From inifile, read in:
# - name of consensus choices, number determined by length
# - names of interpreters (read until done, take length of vector to be the number of names)
# - starting cosmology
# - true cosmology
# - impatient value
#
# Specify how many times you want to run the simulation, set list of random seeds of that length
# For rand in randomSeedList:
# for each consensus in list:
# run consensus record the final results
# --> (tension_metric, final parameter chi2, final data chi2 for all interpreters, number of iterations it took)
# save to file

consensusize = ['ImpatientConsensus', 'ImpatientConsensus']


interpreters = ['SimplePendulumExperiment', 'SimplePendulumExperiment']
number_of_interpreters=len(interpreters)
starting_cosmology = 'CosineCosmology'
true_cosmology = 'DampedDrivenOscillatorCosmology'
truth = getattr(cosmology,true_cosmology)

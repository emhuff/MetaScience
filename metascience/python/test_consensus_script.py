import consensus
import interpret
import numpy as np


cosmologies = [list_of_distinct_cosmologies -- LCDM, wCDM, mNu, etc]

interpreter_A = interpret.CargoCultExperimentInterpreter(parameters = np.zeros(2)+0.)
interpreter_A.chi2 = 1.
interpreter_B = interpret.CargoCultExperimentInterpreter(parameters = np.zeros(2)+1.)
interpreter_B.chi2 = 1.
interpreter_C = interpret.CargoCultExperimentInterpreter(parameters = np.zeros(2)+10.)
interpreter_C.chi2 = 4.

print(interpreter_A.best_fit_cosmological_parameters)

sensible = consensus.SensibleDefaultsConsensus(interpretations = [interpreter_A,interpreter_B,interpreter_C])
sensible.tension_metric()
print(sensible.tm)
print(sensible.is_tension)

sensible.render_judgment()
print(sensible.cosmology_judgment)
print(sensible.systematics_judgment)

'''
How should we update the cosmology?

'''

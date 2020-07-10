import consensus
import interpret
import numpy as np
import cosmology


'''
Things we did July 10
* changed the interpreter to take in a cosmology, that is 'indexed' by a complexity score
* loop over the judgments to pick the new complexity scored cosmology out of a list of possible consensus

Issues right now

* CargoCultCosmology is not initialized with the best fit parameters etc, even though we *think* we did this
* mixing of interpreter/cosmology utilities?
* not sure about importing the .py code cosmology in interpret.py, we didn't do this for the other stuff, but are now in an infinite loop of cosmologies.
'''


cosmologies = [CargoCultCosmology,CargoCultCosmology_Ones, CargoCultCosmology_Tens]
#[StraightLineCosmology,CosineCosmology,TrueCosmology]

index = 0

interpreters[0] = interpret.CargoCultExperimentInterpreter(cosmology = cosmologies[index])
interpreters[0].chi2 = 1.
interpreters[1] = interpret.CargoCultExperimentInterpreter(cosmology = cosmologies[index])
interpreters[1].chi2 = 1.
interpreters[2] = interpret.CargoCultExperimentInterpreter(cosmology = cosmologies[index])
interpreters[2].chi2 = 4.

print(interpreters[0].best_fit_cosmological_parameters)

sensible = consensus.SensibleDefaultsConsensus(interpretations = interpreters)
sensible.tension_metric()
print(sensible.tm)
print(sensible.is_tension)

sensible.render_judgment()
print(sensible.cosmology_judgment)
print(sensible.systematics_judgment)


for i,judgment in enumerate(sensible.cosmology_judgment):
    if judgment == True:
        cosmology[i]=cosmologies[index+1]
        interpreters[i] = interpret.CargoCultExperimentInterpreter(cosmology = cosmologies[i])



'''
How should we update the cosmology?


complexity_order = [cosmo.complexity for cosmo in cosmologies]
# pick cosmology class by complexity:
cosmology_list[where(complexity_order == 2)]


How should we update the interpreter?




'''

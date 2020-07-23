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


cosmologies = [cosmology.CargoCultCosmology(),cosmology.CargoCultCosmology_Ones(), cosmology.CargoCultCosmology_Tens()]
systematics_parameters = [np.arange(2),np.arange(1),np.arange(3)]
#[StraightLineCosmology,CosineCosmology,TrueCosmology]

index = 0
interpreters = []

interpreters.append(interpret.CargoCultExperimentInterpreter(parameters = np.zeros(2),systematics_parameters = systematics_parameters[0],cosmology = cosmologies[index]))
interpreters[0].chi2 = 1.
interpreters.append(interpret.CargoCultExperimentInterpreter(parameters = np.ones(2),cosmology = cosmologies[index],systematics_parameters = systematics_parameters[1]))
interpreters[1].chi2 = 1.
interpreters.append(interpret.CargoCultExperimentInterpreter(parameters = np.ones(2)+10.,cosmology = cosmologies[index],systematics_parameters = systematics_parameters[2]))
interpreters[2].chi2 = 4.

print(interpreters[0].best_fit_cosmological_parameters)

sensible = consensus.SensibleDefaultsConsensus(interpretations = interpreters)
sensible.tension_metric()
print('tension metric = ',sensible.tm)
print('is there tension? ',sensible.is_tension)

sensible.render_judgment()
print('cosmology judgments = ',sensible.cosmology_judgment)
print('systematics judgments = ',sensible.systematics_judgment)


if np.sum(sensible.cosmology_judgment)> 0:
    index = index+1
if np.sum(sensible.systematics_judgement) > 0:
    for i,this_judgment in enumerate(sensible.systematics_judgment):
        if this_judgment:
            new_systematics = np.concatenate(systematics_parameters[i],np.zeros(1))
            systematics_parameters = new_systematics
            interpreters[i] = interpret.CargoCultExperimentInterpreter(parameters = np.zeros(2),systematics_parameters = new_systematics_parameters,cosmology = cosmologies[index])


'''
How should we update the cosmology?


complexity_order = [cosmo.complexity for cosmo in cosmologies]
# pick cosmology class by complexity:
cosmology_list[where(complexity_order == 2)]


How should we update the interpreter?




'''

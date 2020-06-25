import consensus
import interpret
import numpy as np


interpreter_A = interpret.CargoCultExperimentInterpreter()
interpreter_B = interpret.CargoCultExperimentInterpreter()

print(interpreter_A.best_fit_cosmological_parameters)

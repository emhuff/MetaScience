import numpy as np
import matplotlib.pyplot as plt
import experiment
import interpret
import cosmology
import consensus
import dill
import copy
import ipdb
import sys


# 1. Get results file name from argv[1]
# - or get yaml from argv[1] and filename from yaml
# - or make this a function called at end of metascience_simulation script
# 2. Read dill results file
# To read:
'''
with open('data.dill', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = dill.load(f)
'''
# 3. Plot the things

# Parameter history plot
# - in the theme of the Hubble Constant measurements over time
# - use the (to be created) results.xxx_history objects

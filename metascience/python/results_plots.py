import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import experiment
import interpret
import cosmology
import consensus
import dill
import copy
import ipdb
import glob2

names = glob2.glob('*.dill')
#names=['MostlyBetOnThemConsensus-results.dill','ImpatientConsensus-results.dill', 'MostlyBetOnMeConsensus-results.dill']
##filename = sys.argv[1]

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
plt.clf()
colors=['orange', 'black', 'red', 'gray']
labs=names
for fc, filename in enumerate(names):


    with open(filename, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
        data = dill.load(f)

    histories = data.histories

    print('---------')
    # Why are there two histories per file??

    for count, history in enumerate(histories):
        print(f"Count {count} in {filename} \n")

        #

        print('Parameter history: ', history['cosmological_parameter_history'])

        print('Systematics parameter history: ', history['systematics_parameter_history'])
    
    #print('Model history: ', histories[0]['cosmological_model_history'])
    for valc, val in enumerate(histories[0]['cosmological_parameter_history']):
        if histories[0]['cosmological_model_history'][valc]=='Cosine cosmology':
            marker='*'
        else:
            marker='d'
        
        if valc==0:
            plt.scatter(valc,val, marker=marker, color=colors[fc], label=labs[fc])
            plt.scatter(valc,histories[0]['systematics_parameter_history'][valc][0], marker=marker, color=colors[fc+2], label=labs[fc])
        else:   
            plt.scatter(valc,val, marker=marker, color=colors[fc])
            plt.scatter(valc,histories[0]['systematics_parameter_history'][valc][0], marker=marker, color=colors[fc+2])

#    plt.plot(histories[0]['cosmological_parameter_history'], label='%s  - count 0'%filename[:-13])
#    plt.plot(histories[1]['cosmological_parameter_history'], label='%s  - count 1'%filename[:-13])
    #plt.hlines(0.5, 0, len(histories[0]['cosmological_parameter_history']), color='k', linestyles='dashed')
    
plt.xlabel('Iteration number')
plt.ylabel('Cosmological parameter value')
plt.legend(loc='best')
pl.savefig('convergence_summary.png')